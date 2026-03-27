import jax.numpy as jnp
from flax.nnx import Module
from jax import Array
from optax import softmax_cross_entropy_with_integer_labels


def cross_entropy_loss(
    model: Module, batch: Array, labels: Array, ignore_index: int = -100
) -> Array:
    """
    Cross entropy function with mask applied based on ignore_index.
    """
    logits = model(batch)  # (B, T, vocab_size)
    B, T, vocab_size = logits.shape

    mask = labels != ignore_index

    loss = softmax_cross_entropy_with_integer_labels(
        logits.reshape(B * T, vocab_size), labels.reshape(B * T)
    )

    loss = jnp.where(mask.reshape(B * T), loss, 0.0)
    return loss.sum() / mask.sum()
