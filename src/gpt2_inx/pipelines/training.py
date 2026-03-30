import random
from typing import override

import requests
import tiktoken
from flax.nnx import state
from flax.nnx.graphlib import GraphState
from loguru import logger
from tiktoken.core import Encoding
from torch.utils.data import DataLoader, Dataset

from gpt2_inx.configs.hyperparams import GPT2_124M
from gpt2_inx.data.transforms import collate_numpy, format_alpaca
from gpt2_inx.pipelines.model import from_hf
from gpt2_inx.training.loss_functions import cross_entropy_loss
from gpt2_inx.training.trainer import TrainerConfig, train

# -----------------------------
# PyTorch Dataset
# -----------------------------


class InstructionDataset(Dataset[list[int]]):
    def __init__(self, data: list[str], tokenizer: Encoding):
        self.length: int = len(data)
        self.encoded: list[list[int]] = list(map(tokenizer.encode, data))

    @override
    def __getitem__(self, idx: int) -> list[int]:
        return self.encoded[idx]

    def __len__(self):
        return self.length


# -----------------------------
# MetaFlow pipeline
# -----------------------------


def get_instructions(data_url: str) -> list[dict[str, str]]:
    logger.info("Downloading dataset...")
    response = requests.get(data_url)
    response.raise_for_status()

    raw_inx = response.json()
    logger.info(f"Loaded {len(raw_inx)} examples")

    return raw_inx


def process(raw_inx: list[dict[str, str]]) -> list[str]:
    logger.info("Formatting to Alpaca style...")
    return [format_alpaca(x) for x in raw_inx]


def split(
    data: list[str], train_split: float = 0.85, val_split: float = 0.1, seed: int = 42
) -> tuple[list[str], list[str], list[str]]:

    logger.info("Splitting dataset...")

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


def train_model(model_id: str, train_data: list[str], val_data: list[str]) -> GraphState:

    logger.info("Creating PyTorch datasets and dataloaders...")

    tokenizer = tiktoken.get_encoding(model_id)
    config = TrainerConfig(batch_size=8, n_epochs=2, learning_rate=5e-5, weight_decay=0.1)
    num_workers = 0

    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_numpy,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_numpy,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    # setup model and trainer
    hfmodel = model_id
    model = from_hf(GPT2_124M, hfmodel)

    logger.info("Running model {}", hfmodel)
    train(model, train_loader, val_loader, cross_entropy_loss, config)

    return state(model)


def finetune(url: str, model_id: str):
    raw = get_instructions(url)
    inx = process(raw)
    train_data, val_data, _ = split(inx)
    _ = train_model(model_id, train_data, val_data)


def main():
    pass


if __name__ == "__main__":
    main()
