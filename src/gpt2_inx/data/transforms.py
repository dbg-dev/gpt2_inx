from collections.abc import Iterable

import jax.numpy as jnp

# -----------------------------
# Instruction formatting
# There are two types of formatting for instructions:
# Alpaca
# Phi-3
# -----------------------------


def format_alpaca(entry: dict[str, str]):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    response_text = f"\n\n### Response:\n{entry['output']}"

    prompt = instruction_text + input_text + response_text
    return prompt


def collate(
    batch: Iterable[list[int]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    # Find the longest sequence in the batch
    max_len = max(map(len, batch))

    # Pad and prepare inputs and targets
    data: list[list[int]] = []
    labels: list[list[int]] = []

    for item in batch:
        new_item = item.copy()
        pad_len = max_len - len(new_item)
        datum = new_item + [pad_token_id] * pad_len
        label = new_item[1:] + [pad_token_id] + [ignore_index] * pad_len

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            datum = datum[:allowed_max_length]
            label = label[:allowed_max_length]

        data.append(datum)
        labels.append(label)

    return data, labels


def collate_numpy(
    batch: Iterable[list[int]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    data, labels = collate(batch, pad_token_id, ignore_index, allowed_max_length)
    return jnp.array(data), jnp.array(labels)
