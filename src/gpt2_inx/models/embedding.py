from typing import override

from jax.numpy import arange, int32
from flax.nnx import Dropout, Embed, Module, Rngs
from jax import Array

from gpt2_inx.models.params import hyparams


# TODO: check the naming of Learned embedding. This comes from GPT2
class Learned(Module):
    def __init__(self, hps: hyparams, rngs: Rngs, train: bool = False):
        self.tok_embedding: Embed = Embed(hps.vocab_size, hps.embed_dim, rngs=rngs)
        self.pos_embedding: Embed = Embed(hps.ctx_len, hps.embed_dim, rngs=rngs)
        self.dropout: Dropout = Dropout(rate=hps.dropout_rate, rngs=rngs, deterministic=not train)

    @override
    def __call__(self, token_ids: Array) -> Array:
        _, L = token_ids.shape

        tok_emb = self.tok_embedding(token_ids)
        posns = arange(0, L, dtype=int32)
        pos_emb = self.pos_embedding(posns)[None, :, :]

        x = tok_emb + pos_emb
        return self.dropout(x)
