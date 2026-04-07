from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from typing import Any, TypeAlias

import optax
import jax.numpy as jnp

from jax import Array, device_get, device_put
from flax.nnx import Module, Optimizer, Param, jit, value_and_grad
from jax.random import PRNGKey, permutation, split as rnd_split

import wandb
from gpt2_inx.utils import timeit

LossFn: TypeAlias = Callable[[Module, Array, Array], Array]


@dataclass(slots=True)
class TrainerConfig:
    batch_size: int
    n_epochs: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    seq_len: int = 128  # fine to keep for model config/logging if needed


def iter_batches(
    data: tuple[Array, Array],
    batch_size: int,
    *,
    rng: Array | None = None,
    shuffle: bool = False,
):
    inputs, labels = data

    if inputs.shape != labels.shape:
        raise ValueError("inputs and labels must have the same shape")

    if inputs.ndim != 2:
        raise ValueError("inputs and labels must have shape [N, seq]")

    n = inputs.shape[0]

    if shuffle:
        if rng is None:
            raise ValueError("rng must be provided when shuffle=True")
        perm = permutation(rng, n)
        inputs = inputs[perm]
        labels = labels[perm]

    usable = (n // batch_size) * batch_size

    for start in range(0, usable, batch_size):
        end = start + batch_size
        yield inputs[start:end], labels[start:end]


def prefetch_to_device(iterator: Iterator[Any], size: int = 2) -> Generator[Any, Any, None]:
    from collections import deque

    queue = deque[Any]()

    def _put(batch):
        x, y = batch
        return device_put(x), device_put(y)

    it = iter(iterator)

    try:
        for _ in range(size):
            queue.append(_put(next(it)))
    except StopIteration:
        pass

    while queue:
        batch = queue.popleft()
        yield batch
        try:
            queue.append(_put(next(it)))
        except StopIteration:
            pass


def make_train_step(loss_fn: LossFn):
    @jit
    def train_step(
        model: Module,
        optimizer: Optimizer[Any],
        batch: Array,   # [batch, seq]
        labels: Array,  # [batch, seq]
    ) -> tuple[Module, Optimizer[Any], Array]:
        loss, grads = value_and_grad(loss_fn)(model, batch, labels)
        optimizer.update(model, grads)
        return model, optimizer, loss

    return train_step


def make_val_step(loss_fn: LossFn):
    @jit
    def val_step(
        model: Module,
        batch: Array,
        labels: Array,
    ) -> Array:
        return loss_fn(model, batch, labels)

    return val_step


@timeit
def train(
    model: Module,
    train_data: tuple[Array, Array],
    val_data: tuple[Array, Array],
    loss_fn: LossFn,
    config: TrainerConfig,
):
    optimizer = Optimizer[Any](
        model,
        optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
        wrt=Param,
    )

    train_step = make_train_step(loss_fn)
    val_step = make_val_step(loss_fn)

    run = wandb.init(
        entity="minky-raccoon-me",
        project="my-gpt2-sft-project",
        config={
            "learning_rate": config.learning_rate,
            "architecture": "GPT2",
            "dataset": "Rabast instructions",
            "epochs": config.n_epochs,
            "seq_len": config.seq_len,
            "batch_size": config.batch_size,
        },
    )

    key = PRNGKey(0)

    for epoch in range(config.n_epochs):
        key, subkey = rnd_split(key)

        train_iter = prefetch_to_device(
            iter_batches(
                train_data,
                config.batch_size,
                rng=subkey,
                shuffle=True,
            ),
            size=2,
        )

        val_iter = prefetch_to_device(
            iter_batches(
                val_data,
                config.batch_size,
                shuffle=False,
            ),
            size=2,
        )

        model.train()
        train_losses = []
        for batch_x, batch_y in train_iter:
            model, optimizer, loss = train_step(model, optimizer, batch_x, batch_y)
            train_losses.append(loss)

        model.eval()
        val_losses = []
        for batch_x, batch_y in val_iter:
            loss = val_step(model, batch_x, batch_y)
            val_losses.append(loss)

        train_loss = float(device_get(jnp.mean(jnp.stack(train_losses))))
        val_loss = float(device_get(jnp.mean(jnp.stack(val_losses))))

        run.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

    run.finish()