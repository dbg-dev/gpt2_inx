from typing import Any, Hashable, TypeAlias

from flax.nnx import Module, Rngs, merge, split, state, to_pure_dict
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import Array, random
from jax.numpy import array
from loguru import logger

from gpt2_inx.configs.modelmaps import hfgpt2_to_local
from gpt2_inx.models.gpt2 import GPT2
from gpt2_inx.models.params import hyparams

from torch import Tensor

ParamKey: TypeAlias = tuple[Hashable, ...]


def validate_mapping_sets(
    src_params: dict[str, Any],
    target_params: dict[Any, Any],
    mapping: dict[Any, str],
    *,
    check_shapes: bool = True,
) -> None:
    """
    A set-based validation of a mapping from HF → Flax/NNX parameter keys.
    Ensures:
      - every HF key referenced by mapping exists
      - every model key referenced by mapping exists
      - reports unused HF keys
      - optional: checks shape compatibility
    """

    src_keys = set(src_params.keys())  # strings
    target_keys = set(target_params.keys())  # tuple paths
    mapped_hf = set(mapping.values())
    mapped_model = set(mapping.keys())

    errors = []

    # --- 1. Mapped HF keys must exist in source ---
    missing_hf = mapped_hf - src_keys
    if missing_hf:
        errors.append(
            "src doesn't have mappping:\n  "
            + "\n  ".join(repr(k) for k in sorted(missing_hf))
        )

    # --- 2. Mapped model keys must exist in model params ---
    missing_model = mapped_model - target_keys
    if missing_model:
        errors.append(
            "Mapped model keys not present in target_params:\n  "
            + "\n  ".join(repr(k) for k in sorted(missing_model))
        )

    # --- 3. Warn about unused HF parameters ---
    unused_hf = src_keys - mapped_hf
    if unused_hf:
        logger.warning(
            f"{len(unused_hf)} HF parameters are unused by the mapping:\n  "
            + ", ".join(sorted(unused_hf))
        )

    # --- 4. Optional: shape validation for overlapping keys ---
    if check_shapes:
        for mkey, skey in mapping.items():
            if mkey not in target_params or skey not in src_params:
                continue

            src_shape = tuple(src_params[skey].shape)
            model_shape = tuple(target_params[mkey].shape)

            if src_shape != model_shape:
                errors.append(
                    f"Shape mismatch for mapping:\n"
                    f"  model key {mkey!r} shape={model_shape}\n"
                    f"  HF key    {skey!r} shape={src_shape}"
                )

    # --- 5. Final result ---
    if errors:
        msg = "\n\n".join(errors)
        raise KeyError(f"Mapping validation failed:\n\n{msg}")

    logger.info("Set-based mapping validation passed ✓")


def map_params(
    src_params: dict[str, Tensor],
    model_params: dict[ParamKey, Array],
    mapping: dict[Any, str],
) -> dict[Any, Array]:

    def get_src(k: Any, v: Array) -> Array:
        ptr = mapping.get(k, None)

        if ptr is None:
            if not "rngs" in k:
                logger.warning(f"No mapping for key {k!r}, keeping initialised value")
            return v
        if ptr == "lm_head.weight":
            return array(src_params[ptr].cpu().numpy().T)

        return array(src_params[ptr].cpu().numpy())

    return {
        k: get_src(k, v) 
        for k, v 
        in model_params.items()
    }


def model_mapper(src, target: Module, mod_map, num_layers: int) -> Module:
    graphdef, target_state = split(target)
    src_dict = src.state_dict()
    target_dict: dict[ParamKey, Array] = flatten_dict(to_pure_dict(target_state))

    mapping = mod_map(num_layers)
    # validate_mapping_sets(src_dict, target_dict, mapping)
    mapped = map_params(src_dict, target_dict, mapping)
    mapped_state = state(unflatten_dict(mapped))

    return merge(graphdef, mapped_state)


def from_hf(cfg: hyparams, hf_model_id: str) -> Module:
    from transformers import AutoModelForCausalLM

    model = GPT2(cfg, Rngs(random.key(0)))  # weights overwritten below
    hf = AutoModelForCausalLM.from_pretrained(hf_model_id)

    return model_mapper(hf, model, hfgpt2_to_local, cfg.num_layers)
