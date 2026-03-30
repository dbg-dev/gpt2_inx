from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optax
from flax.nnx import Module, Optimizer, Param, jit, value_and_grad
from jax import Array, device_count, device_put, process_index, lax, devices, default_backend
from torch.utils.data import DataLoader

import wandb
import numpy as np

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils


@dataclass(slots=True)
class TrainerConfig:
    batch_size: int
    n_epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.1


def make_train_step(loss_fn: Callable[[Module, Array, Array], Array]):
    @jit
    def train_step(model: Module, optimizer: Optimizer[Any], batch: Array, labels: Array):
        loss, grads = value_and_grad(loss_fn)(model, batch, labels)
        optimizer.update(model, grads)
        return loss

    return train_step


def make_val_step(loss_fn: Callable[[Module, Array, Array], Array]):
    @jit
    def val_step(model: Module, batch: Array, labels: Array) -> Array:
        return loss_fn(model, batch, labels)

    return val_step


def train(
    model: Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    loss_fn: Callable[[Module, Array, Array], Array],
    config: TrainerConfig,
):

    # --- mesh setup ---
    devices = mesh_utils.create_device_mesh((device_count(),))
    mesh = Mesh(devices, axis_names=("batch",))
    data_sharding = NamedSharding(mesh, P("batch"))
    replicated_sharding = NamedSharding(mesh, P())

    def shard_batch(batch: Any, labels: Any):
        batch = np.array(batch)
        labels = np.array(labels)
        if device_count() > 1:
            return (
                device_put(batch, data_sharding),
                device_put(labels, data_sharding),
            )
        # single device — just move to device normally
        return device_put(batch), device_put(labels)

    optimizer = Optimizer[Any](
        model,
        optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay),
        wrt=Param,
    )

    # Explicitly replicate model+optimizer state across all TPU devices
    device_put(optimizer, replicated_sharding)

    train_step = make_train_step(loss_fn)
    val_step = make_val_step(loss_fn)

    # --- wandb: main process only ---
    if process_index() == 0:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="minky-raccoon-me",
            # Set the wandb project where this run will be logged.
            project="my-gpt2-sft-project",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": config.learning_rate,
                "architecture": "GPT2",
                "dataset": "Rabast instructions",
                "epochs": config.n_epochs,
            },
        )

    for epoch in range(config.n_epochs):
        # train
        train_loss, train_samples = 0.0, 0
        for batch, labels in train_loader:
            batch, labels = shard_batch(batch, labels)
            train_loss += train_step(model, optimizer, batch, labels) * len(batch)
            train_samples += len(batch)

        # val
        val_loss, val_samples = 0.0, 0
        for batch, labels in val_loader:
            batch, labels = shard_batch(batch, labels)
            val_loss += val_step(model, batch, labels) * len(batch)
            val_samples += len(batch)

        if process_index() == 0:
            run.log(
                {"acc": (train_loss / train_samples), "loss": (val_loss / val_samples)}
            )

    if process_index() == 0:
        # Finish the run and upload any remaining data.
        run.finish()
