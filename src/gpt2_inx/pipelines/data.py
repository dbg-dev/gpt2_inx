import random

from typing import Any
import requests
import tiktoken
from loguru import logger
from jax.numpy import asarray, int32

type Sample = tuple[list[int], list[int]]

# -----------------------------
# Pipeline
# -----------------------------


def get_instructions(data_url: str) -> list[dict[str, str]]:
    logger.info("Downloading dataset...")
    response = requests.get(data_url)
    response.raise_for_status()

    raw_inx = response.json()
    logger.info(f"Loaded {len(raw_inx)} examples")

    return raw_inx


# -----------------------------
# Instruction formatting
# There are two types of formatting for instructions:
# Alpaca
# Phi-3
# -----------------------------


def format_alpaca(entry: dict[str, str]) -> tuple[str, str]:
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    response_text = f"\n\n### Response:\n{entry['output']}"

    prompt = instruction_text + input_text
    return prompt, response_text


def format_to_alpaca(raw_inx: list[dict[str, str]]) -> list[tuple[str, str]]:
    logger.info("Formatting to Alpaca style...")
    return [format_alpaca(x) for x in raw_inx]


def pad(
    tokens: list[int],
    seq_len: int,
    pad_token_id: int = 50256,
    ignore_index: int = -100,
) -> Sample:
    # ---- inputs ----
    input = tokens[:seq_len]
    if len(input) < seq_len:
        input = input + [pad_token_id] * (seq_len - len(input))

    # ---- labels ----
    label = tokens[1:seq_len] + [pad_token_id]
    if len(label) < seq_len:
        label = label + [ignore_index] * (seq_len - len(label))

    return input, label


def split(
    data: list[Any], 
    train_split: float = 0.85, 
    val_split: float = 0.1, 
    seed: int | None = 42
) -> tuple[list[Any], list[Any], list[Any]]:

    logger.info("Splitting dataset...")

    if seed is not None: 
        random.seed(seed)
        random.shuffle(data)

    n = len(data)
    train_size = int(n * train_split)
    val_size = int(n * val_split)

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    logger.debug(f"Train: {len(train_data)}")
    logger.debug(f"Val:   {len(val_data)}")
    logger.debug(f"Test:  {len(test_data)}")

    return train_data, val_data, test_data


def to_jax(dataset: Sample):
    inputs, labels = [asarray(x, int32) for x in zip(*dataset)]
    return inputs, labels


def prep_dataset(ds: list[dict[str, str]], tokenizer: tiktoken.core.Encoding):
    """
    turns dataset into alpaca format, encodes, splits into input and target, then pads to same length
    """
    encoded = [
        tokenizer.encode("".join(format_alpaca(d)))
        for d in ds
    ]
    max_len = max(map(len, encoded)) + 1
    xys = ([pad(e, max_len) for e in encoded])
    return to_jax(xys)




def prep_test_inxs(
    instructions: list[dict[str, str]], 
    tokenizer: tiktoken.core.Encoding
):
    """
    Encodes alpaca instructions without responses to use for testing
    return is encoded instruction plus expected response string
    """
    return [format_alpaca(i) for i in instructions]



def prepare(url: str, tokenizer: tiktoken.core.Encoding):
    training, test, validation = split(get_instructions(url))

    train_ds, val_ds = [prep_dataset(d, tokenizer) for d in [training, validation]]
    test_prompts = prep_test_inxs(test, tokenizer)

    return train_ds, val_ds, test_prompts


def main():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    tokenizer = tiktoken.get_encoding("gpt2")
    _ = prepare(url, tokenizer)




if __name__ == "__main__":
    main()
