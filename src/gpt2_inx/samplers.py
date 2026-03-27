from jax._src.basearray import Array


import flax.nnx as nnx
import jax
from jax.numpy import arange, argmax, concat, int32, take, zeros

from gpt2_inx.utils import timeit

"""
This code needs some notes.
If you write this naively with a Python `for` loop like this it will be very slow.
```
@nnx.jit
def sample_greedy(model: nnx.Module, rngs: nnx.Rngs, input_ids: jax.Array, max_new_tokens: int=10):
    for _ in range(max_new_tokens):
        logits = model(input_ids, rngs)
        next_token = argmax(logits[:, -1, :], axis=-1)  # last position only
        input_ids = concat([input_ids, next_token[:, None]], axis=1)
    return input_ids
```
(I think) This is because although the inner code is run in XLA, but it needs to come out to Python to iterate.
To solve this need to use `jax.lax.scan` and define an inner function to create a for loop in XLA.
The other step is that can't use dynamic array because XLA needs static arrays.
Hence we define everything up front and update each value in place.

This took about a day of digging around to get to this 20 line solution. The result was a 4x speed-up on CPU.

Useful resources:

https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
https://github.com/cgarciae/nanoGPT-jax/blob/master/model.py - this is FLax linen, but provided the `generate` function provided the solution.
https://apxml.com/courses/advanced-jax/chapter-1-advanced-jax-transformations-control-flow/mastering-lax-scan - this is a nice explanation.
"""


@timeit
@nnx.jit
def sample_greedy(model: nnx.Module, x: jax.Array, n: int = 10):
    """
    Does something
    """
    B, T = x.shape
    padding = zeros((B, n), dtype=int32)
    tokens = concat([x, padding], axis=-1)
    idxs = arange(T, T + n)

    def scan_f(carry: jax.Array, i: int) -> tuple[Array, Array]:
        logits = model(carry)
        logits = take(logits, indices=i-1, axis=1)
        next_token = argmax(logits, axis=-1)
        # Update running context in fixed position
        carry = carry.at[:, i].set(next_token)

        return carry, next_token

    tokens, _ = jax.lax.scan(scan_f, tokens, idxs)
    return tokens
