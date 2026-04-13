from jax.numpy import where
from jax import Array
from optax import softmax_cross_entropy_with_integer_labels


def cross_entropy_loss(
    logits: Array,
    labels: Array,
    ignore_index: int = -100,
) -> Array:
    """
    Cross entropy loss with proper masking for ignore_index.

    Fixes:
    - Replaces invalid labels before CE (Optax requires valid indices)
    - Applies mask AFTER CE
    - Avoids divide-by-zero
    """

     # (B, T, vocab_size)
    B, T, vocab_size = logits.shape

    # Flatten for Optax
    logits_flat = logits.reshape(B * T, vocab_size)
    labels_flat = labels.reshape(B * T)

    # Create mask: True where label is valid
    mask = labels_flat != ignore_index

    # Replace ignored labels with a safe dummy class (e.g. 0)
    safe_labels = where(mask, labels_flat, 0)

    # Compute per-token CE
    loss = softmax_cross_entropy_with_integer_labels(
        logits_flat,
        safe_labels,
    )

    # Zero out ignored positions
    loss = where(mask, loss, 0.0)

    # Normalize safely
    denom = mask.sum()
    denom = where(denom == 0, 1, denom)  # avoid NaN

    return loss.sum() / denom
