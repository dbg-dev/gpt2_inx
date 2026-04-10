from typing import override

from flax.nnx import Dropout, LayerNorm, Module, Rngs, gelu
from flax.nnx.helpers import List, Sequential
from flax.nnx.nn.linear import Linear
from jax import Array

from gpt2_inx.models.attention import MultiHeadSelfAttention
from gpt2_inx.models.embedding import Learned
from gpt2_inx.models.params import hyparams


class MLP(Module):
    def __init__(self, hps: hyparams, rngs: Rngs):
        self.mlp: Sequential = Sequential(
            Linear(hps.embed_dim, hps.ff_hidden_dim, use_bias=hps.use_bias, rngs=rngs),
            gelu,
            Linear(hps.ff_hidden_dim, hps.embed_dim, use_bias=hps.use_bias, rngs=rngs),
        )

    @override
    def __call__(self, x: Array):
        return self.mlp(x)


class TransformerBlock(Module):
    def __init__(self, hps: hyparams, rngs: Rngs):
        self.mha: Sequential = Sequential(
            LayerNorm(hps.embed_dim, use_bias=hps.use_bias, rngs=rngs, epsilon=1e-5),
            MultiHeadSelfAttention(hps, rngs=rngs)
        )
        self.mha_dp: Dropout = Dropout(hps.dropout_rate)
        
        self.ff: Sequential = Sequential(
            LayerNorm(hps.embed_dim, use_bias=hps.use_bias, rngs=rngs, epsilon=1e-5),
            MLP(hps, rngs=rngs)
        )
        self.ff_dp: Dropout = Dropout(hps.dropout_rate)

    @override
    def __call__(self, x: Array, *, rngs: Rngs | None = None):
        mha = self.mha(x, rngs=rngs)
        x = x + self.mha_dp(mha, rngs=rngs)
        ff = self.ff(x)
        x = x + self.ff_dp(ff, rngs=rngs)
        return x


class GPT2(Module):
    def __init__(self, hps: hyparams, rngs: Rngs):
        self.embed: Learned = Learned(hps, rngs)
        self.blocks: List[TransformerBlock] = List(
            [TransformerBlock(hps, rngs) for _ in range(hps.num_layers)]
        )
        self.layernorm: LayerNorm = LayerNorm(
            hps.embed_dim, use_bias=hps.use_bias, rngs=rngs
        )
        self.lm_head: Linear = Linear(
            hps.embed_dim, hps.vocab_size, use_bias=False, rngs=rngs
        )

    @override
    def __call__(self, token_ids: Array, *, rngs: Rngs | None = None):
        x = self.embed(token_ids, rngs=rngs)
        for block in self.blocks:
            x = block(x, rngs=rngs)
        x = self.layernorm(x)
        logits = self.lm_head(x)
        return logits
