from typing import override

from flax.nnx import Module, Dropout, Rngs, softmax
from flax.nnx.nn.linear import Linear
from jax import Array
from jax.numpy import inf, split, sqrt, tri, where

from gpt2_inx.models.params import hyparams


class MultiHeadSelfAttention(Module):
    def __init__(self, hps: hyparams, rngs: Rngs) -> None:
        self.embed_dim: int = hps.embed_dim
        self.num_heads: int = hps.num_heads
        self.head_dim: int = hps.embed_dim // hps.num_heads

        self.qkv: Linear = Linear(
            self.embed_dim,
            self.embed_dim * 3,  # leave hardcoded because always 3: query, key and value
            use_bias=hps.use_bias,
            rngs=rngs,
        )
        self.lin: Linear = Linear(
            self.embed_dim, self.embed_dim, use_bias=hps.use_bias, rngs=rngs
        )
        self.dp: Dropout = Dropout(rate=hps.dropout_rate, rngs=rngs)

    @override
    def __call__(self, x: Array) -> Array:
        B, L, _ = x.shape

        def split_heads(t: Array) -> Array:
            batch_size, num_tokens = t.shape[0], t.shape[1]
            return t.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )

        qkv = split(self.qkv(x), 3, axis=-1)

        q, k, v = map(split_heads, qkv)

        mask = tri(L, dtype=int)
        qk = q @ k.swapaxes(-1, -2)
        attn = qk / sqrt(k.shape[-1])
        attn = where(mask == 0, -inf, attn)
        attn = softmax(attn, axis=-1) @ v
        z = attn.transpose(0, 2, 1, 3).reshape(B, L, self.embed_dim)
        return self.dp(self.lin(z))
