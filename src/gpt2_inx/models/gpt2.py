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
            Dropout(rate=hps.dropout_rate, rngs=rngs),
        )

    @override
    def __call__(self, x: Array):
        # return pipe(x, self.ln1, nnx.gelu, self.ln2, self.dp)
        return self.mlp(x)


class TransformerBlock(Module):
    def __init__(self, hps: hyparams, rngs: Rngs):
        self.att: Sequential = Sequential(
            LayerNorm(hps.embed_dim, use_bias=hps.use_bias, rngs=rngs, epsilon=1e-5),
            MultiHeadSelfAttention(hps, rngs),
        )
        self.ff: Sequential = Sequential(
            LayerNorm(hps.embed_dim, use_bias=hps.use_bias, rngs=rngs, epsilon=1e-5), MLP(hps, rngs)
        )

    @override
    def __call__(self, x: Array):
        x = x + self.att(x)
        x = x + self.ff(x)
        return x


class GPT2(Module):
    def __init__(self, hps: hyparams, rngs: Rngs):
        self.embed: Learned = Learned(hps, rngs)
        self.blocks: List[TransformerBlock] = List(
            [TransformerBlock(hps, rngs) for _ in range(hps.num_layers)]
        )
        self.final_norm: LayerNorm = LayerNorm(hps.embed_dim, use_bias=hps.use_bias, rngs=rngs)
        self.lm_head: Linear = Linear(hps.embed_dim, hps.vocab_size, use_bias=False, rngs=rngs) 

    @override
    def __call__(self, token_ids: Array):
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
