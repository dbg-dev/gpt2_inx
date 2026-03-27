from typing import Any

from flax.nnx import Module, Rngs, merge, split, state, to_pure_dict
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import Array, random
from loguru import logger

from gpt2_inx.configs.modelmaps import hfgpt2_to_local
from gpt2_inx.models.gpt2 import GPT2
from gpt2_inx.models.params import hyparams


def map_params(
    src_params: dict[str, Array], model_params: dict[Any, Array], mapping: dict[Any, str]
) -> dict[Any, Array]:

    def get_src(k: Any, v: Array) -> Array:
        ptr = mapping.get(k, None)
        if ptr is None:
            return v
        iskernel = k[-1].lower() == "kernel"
        arr = src_params[ptr]
        out = arr.T if iskernel else arr

        if out.shape != v.shape:
            # TODO: return error if parameter import does not match on shape.
            logger.error("Parameter import does not match on shape")

        return out

    mapped = {k: get_src(k, v) for k, v in model_params.items()}

    return mapped


def model_mapper(src, target: Module, mod_map, num_layers: int) -> Module:
    graphdef, target_state = split(target)
    src_params = flatten_dict(src.params, sep=".")
    target_params = flatten_dict(to_pure_dict(target_state))

    mapping = mod_map(num_layers)
    mapped = map_params(src_params, target_params, mapping)
    mapped_state = state(unflatten_dict(mapped))

    return merge(graphdef, mapped_state)


def from_hf(cfg: hyparams, hf_model: str) -> Module:
    # from transformers import AutoModelForCausalLM as hfmodel
    from transformers import FlaxGPT2LMHeadModel as hfmodel

    key = random.key(0)
    model = GPT2(cfg, Rngs(key))
    hfmodel = hfmodel.from_pretrained(hf_model)

    return model_mapper(hfmodel, model, hfgpt2_to_local, cfg.num_layers)
