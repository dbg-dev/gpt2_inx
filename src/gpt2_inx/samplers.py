
# import flax.nnx as nnx
# import jax
from jax.numpy import ndarray, where, inf
from jax import Array
from functools import partial 
from jax.lax import top_k
from jax.random import categorical
from jax import jit


@partial(jit, static_argnames=("k",))
def topk_sample(logits: ndarray, k: int, key: int) -> Array:
    """
    Top-k sampling from a logit distribution.

    Args:
        logits: shape (vocab_size,) — raw (unnormalized) logits
        k:      number of top candidates to keep (static, traced at compile time)
        key:    JAX PRNG key

    Returns:
        sampled token index, shape ()
    """
    # 1. Find the k-th largest logit value (scalar threshold)
    top_values, _ = top_k(logits, k)
    # top_values is (..., k) — take the last column and keep dims
    # so threshold broadcasts correctly against logits
    threshold = top_values[..., -1:]               # (..., 1) instead of (...)

    # 2. Mask out everything below the threshold
    masked_logits = where(logits >= threshold, logits, -inf)

    # 3. Convert to probabilities and sample
    return categorical(key, masked_logits)



