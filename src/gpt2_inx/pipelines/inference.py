from typing import Any
from gpt2_inx.utils import timeit
from functools import partial 
from flax.nnx import Module, GraphDef, GraphState, merge, split
from jax import Array, jit
from jax.numpy import int32, zeros, concat, where, bool_
from jax.lax import scan
from jax.random import PRNGKey, split as rndm_split

from gpt2_inx.samplers import topk_sample


EOS_TOKEN_ID = 50256

@partial(jit, static_argnames=("graphdef", "k", "max_new_tokens"))
def _generate_loop(
    graphdef: GraphDef[Any], 
    state: GraphState, 
    prompt_ids: Array, 
    k:int,                  # set sample size 
    max_new_tokens: int,
    key: int                # random split
) -> tuple[Array, Any]:
    # preallocate a fixed-length buffer
    zeros_pad = zeros((1, max_new_tokens), dtype=int32)
    tokens = concat([prompt_ids, zeros_pad], axis=1)


    def step(carry, _):
        tokens, pos, key, done = carry
        key, subkey = rndm_split(key)

        model  = merge(graphdef, state)
        logits = model(tokens)                  # (1, max_len, vocab_size)
        last_logits = logits[0, pos - 1, :]     # (vocab_size,)
        next_token = topk_sample(last_logits, k=k, key=subkey)

        # If already done, emit EOS and don't update tokens
        next_token = where(done, EOS_TOKEN_ID, next_token)
        tokens = tokens.at[0, pos].set(next_token)

        done = done | (next_token == EOS_TOKEN_ID)
        return (tokens, pos + 1, key, done), next_token


    start_pos = prompt_ids.shape[1]
    (tokens, _, _, _), new_tokens = scan(
        step,
        (tokens, start_pos, key, bool_(False)),
        None,
        length=max_new_tokens,
    )

    return tokens, new_tokens



@timeit
def generate(
    model: Module, 
    prompt_ids: Array, 
    max_new_tokens: int, 
    k:int = 50, 
    seed:int =  42
):
    key = PRNGKey(seed)
    graphdef, state = split(model)
    tokens, _ = _generate_loop(
        graphdef, 
        state,
        prompt_ids,
        k, max_new_tokens, key
    )
    return tokens[0]   # drop batch dim -> (max_len,)