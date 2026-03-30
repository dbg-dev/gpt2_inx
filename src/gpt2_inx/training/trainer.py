from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optax
from flax.nnx import Module, Optimizer, Param, jit, value_and_grad, split, merge
from jax.numpy import array
from jax import Array, device_put, devices
from torch.utils.data import DataLoader

import wandb



@dataclass(slots=True)
class TrainerConfig:
    batch_size: int
    n_epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.1


def train(
    model: Module, 
    train_loader: DataLoader[Any], 
    val_loader: DataLoader[Any],
    loss_fn: Callable[[Module, Array, Array], Array], 
    config: TrainerConfig
):
    # Move model to GPU
    graphdef, state = split(model)
    state = device_put(state, devices()[0])
    model = merge(graphdef, state)

    optimizer = Optimizer(
        model,
        optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay),
        wrt=Param
    )

    @jit
    def train_step(model: Module, optimizer: Optimizer[Any], batch: Array, labels: Array):
        loss, grads = value_and_grad(loss_fn)(model, batch, labels)
        optimizer.update(model, grads)
        return loss

    @jit
    def val_step(model: Module, batch: Array, labels: Array):
        return loss_fn(model, batch, labels)

    for epoch in range(config.n_epochs):
        train_loss, train_samples = 0.0, 0
        for batch, labels in train_loader:
            batch = device_put(array(batch), devices()[0])
            labels = device_put(array(labels), devices()[0])
            loss = train_step(model, optimizer, batch, labels)
            train_loss += float(loss) * len(batch)
            train_samples += len(batch)

        val_loss, val_samples = 0.0, 0
        for batch, labels in val_loader:
            batch = device_put(array(batch), devices()[0])
            labels = device_put(array(labels), devices()[0])
            loss = val_step(model, batch, labels)
            val_loss += float(loss) * len(batch)
            val_samples += len(batch)

        print(f"Epoch {epoch+1}: train_loss={train_loss/train_samples:.4f} val_loss={val_loss/val_samples:.4f}")