from typing import Any

from flax.nnx import Module, Rngs, merge, split, state, to_pure_dict
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import Array, random
from jax.numpy import array
from loguru import logger

from gpt2_inx.configs.modelmaps import hfgpt2_to_local
from gpt2_inx.models.gpt2 import GPT2
from gpt2_inx.models.params import hyparams

from torch import Tensor


def map_params(
    src_params: dict[str, Tensor],
    model_params: dict[Any, Array],
    mapping: dict[Any, str],
) -> dict[Any, Array]:

    def get_src(k: Any, v: Array) -> Array:
        ptr = mapping.get(k, None)
        if ptr is None:
            logger.warning(f"No mapping for key {k!r}, keeping initialised value")
            return v
        return array(src_params[ptr].detach().numpy()) 

    return {
        k: get_src(k, v) 
        for k, v 
        in model_params.items()
    }


def model_mapper(src, target: Module, mod_map, num_layers: int) -> Module:
    graphdef, target_state = split(target)
    src_dict = src.state_dict()
    target_dict = flatten_dict(to_pure_dict(target_state))

    mapping = mod_map(num_layers)
    mapped = map_params(src_dict, target_dict, mapping)
    mapped_state = state(unflatten_dict(mapped))

    return merge(graphdef, mapped_state)


def from_hf(cfg: hyparams, hf_model_id: str) -> Module:
    from transformers import AutoModelForCausalLM

    model = GPT2(cfg, Rngs(random.key(0)))  # weights overwritten below
    hf = AutoModelForCausalLM.from_pretrained(hf_model_id)

    return model_mapper(hf, model, hfgpt2_to_local, cfg.num_layers)
