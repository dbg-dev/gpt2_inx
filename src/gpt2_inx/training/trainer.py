from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

import optax
from jax import Array, device_get
from jax.numpy import mean
from flax.nnx import Module, Optimizer, Param, jit, value_and_grad, scan, Carry
from jax.random import PRNGKey, permutation, split as rnd_split

import wandb

LossFn: TypeAlias = Callable[[Module, Array, Array], Array]


@dataclass(slots=True)
class TrainerConfig:
    batch_size: int
    n_epochs: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    seq_len: int = 128


def make_train_epoch(loss_fn: LossFn):
    @jit
    def train_epoch(
        model: Module,
        optimizer: Optimizer[Any],
        batches: Array,  # [num_batches, batch, seq]
        labels: Array,   # [num_batches, batch, seq]
    ) -> tuple[Any | Module, Any | Optimizer[Any], Array]:
        @scan(
            in_axes=(Carry, 0, 0), 
            out_axes=(Carry, 0)
        )
        def scan_step(
            carry: tuple[Module, Optimizer[Any]], 
            batch: Array, 
            label: Array
        ) -> tuple[tuple[Module, Optimizer[Any]], Any]:
            model, optimizer = carry

            loss, grads = value_and_grad(loss_fn)(model, batch, label)
            optimizer.update(model, grads)
            return (model, optimizer), loss

        (model, optimizer), losses = scan_step((model, optimizer), batches, labels)
        return model, optimizer, mean(losses)

    return train_epoch


def make_val_epoch(loss_fn: LossFn):
    @jit
    def val_epoch(
        model: Module,
        batches: Array,
        labels: Array,
    ):
        @scan(
            in_axes=(Carry, 0, 0), 
            out_axes=(Carry, 0)
        )
        def scan_step(model: Module, batch: Array, label: Array):
            loss = loss_fn(model, batch, label)
            return model, loss

        model, losses = scan_step(model, batches, labels)
        return mean(losses)

    return val_epoch


def shuffle_and_batch(
    data: tuple[Array, Array],
    cfg: TrainerConfig,
    rng: Array | None = None,
    shuffle: bool = True
) -> tuple[Array, Array]:
    inputs, labels = data

    if inputs.shape != labels.shape:
        raise ValueError("inputs and labels must have the same shape")

    if inputs.ndim != 2:
        raise ValueError("inputs and labels must have shape [N, seq_len]")

    n, _ = inputs.shape
    usable = (n // cfg.batch_size) * cfg.batch_size

    if shuffle:
        if rng is None:
            raise ValueError("rng must be provided when shuffle=True")
        perm = permutation(rng, n)
        inputs = inputs[perm]
        labels = labels[perm]

    # drop remainder and batch
    inputs = inputs[:usable].reshape(-1, cfg.batch_size, cfg.seq_len)
    labels = labels[:usable].reshape(-1, cfg.batch_size, cfg.seq_len)

    return inputs, labels

def train(
    model: Module,
    train_data: tuple[Array,Array],
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

    train_epoch = make_train_epoch(loss_fn)
    val_epoch = make_val_epoch(loss_fn)

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
    vxs, vys = shuffle_and_batch(val_data, config, None, False)

    key = PRNGKey(0)
    for epoch in range(config.n_epochs):
        key, subkey = rnd_split(key)
        txs, tys = shuffle_and_batch(train_data, config, subkey, True)

        # model.train()
        model, optimizer, train_loss = train_epoch(
            model, optimizer, txs, tys
        )

        # model.eval()
        val_loss = val_epoch(model, vxs, vys)
        
        train_loss = float(device_get(train_loss))
        val_loss = float(device_get(val_loss))

        run.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

    run.finish()
