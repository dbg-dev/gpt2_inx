from collections.abc import Callable
from typing import Any, TypeAlias
from pathlib import Path
from loguru import logger


import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from grain import DataLoader
from flax import struct
from flax import nnx
from jax import Array, device_get

from gpt2_inx.configuration import RuntimeConfig, CheckpointConfig



LossFn: TypeAlias = Callable[[Array, Array], Array]
EvalFn: TypeAlias = Callable[[Array, Array], dict[str, Array]]


@struct.dataclass # Turns the class into a pyTree and makes it immutable
class TrainState:
    graphdef: nnx.GraphDef[Any]
    params: nnx.State
    opt_state: optax.OptState
    rng_key: jax.Array
    step: Array  # jnp.int32 scalar


# ------------------
# Checkpointing
# ------------------

class CheckpointIO:
    def __init__(self, config: CheckpointConfig):
        self.config: CheckpointConfig = config
        self.root: Path = Path(config.dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def payload(
        self,
        state: TrainState,
        runtime_config: RuntimeConfig | None = None,
    ) -> dict[str, Any]:
        payload = {
            "graphdef": state.graphdef,
            "params": state.params,
            "opt_state": state.opt_state,
            "rng_key": state.rng_key,
            "step": state.step,
        }
        if runtime_config is not None:
            payload["runtime_config"] = runtime_config.model_dump()
        return payload

    def save(
        self,
        state: TrainState,
        runtime_config: RuntimeConfig | None = None,
    ) -> Path:
        step = int(device_get(state.step))
        path = self.root / f"step_{step:08d}"

        with ocp.StandardCheckpointer() as ckptr:
            ckptr.save(path, self.payload(state, runtime_config))

        return path

    def restore(
        self,
        checkpoint_path: str | Path,
        abstract_state: TrainState,
    ) -> TrainState:
        path = Path(checkpoint_path)
        abstract_payload = {
            "graphdef": abstract_state.graphdef,
            "params": abstract_state.params,
            "opt_state": abstract_state.opt_state,
            "rng_key": abstract_state.rng_key,
            "step": abstract_state.step,
        }

        with ocp.StandardCheckpointer() as ckptr:
            restored = ckptr.restore(path, abstract_payload)

        return TrainState(
            graphdef=restored["graphdef"],
            params=restored["params"],
            opt_state=restored["opt_state"],
            rng_key=restored["rng_key"],
            step=restored["step"],
        )


# ------------------
# Trainer
# ------------------

def build_lr_schedule(
    *,
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    min_learning_rate: float,
) -> optax.Schedule:
    warmup_steps = max(1, warmup_steps)
    total_steps = max(1, total_steps)

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=min_learning_rate,
    )


def build_tx(
    config: RuntimeConfig,
    total_steps: int,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    lr_schedule = build_lr_schedule(
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        min_learning_rate=config.min_learning_rate,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
        ),
    )
    return tx, lr_schedule


def build_train_state(
    model: nnx.Module,
    tx: optax.GradientTransformation,
    seed: int = 0
) -> TrainState:
    graphdef, params = nnx.split(model, nnx.Param)
    opt_state = tx.init(params)
    return TrainState(
        graphdef=graphdef,
        params=params,
        opt_state=opt_state,
        rng_key=jax.random.key(seed),
        step=jnp.array(0, dtype=jnp.int32),
    )


def materialize_model(state: TrainState) -> nnx.Module:
    model = nnx.merge(state.graphdef, state.params)
    model.eval()
    return model


# ------------------
# Train and eval builders
# ------------------


def make_train_step(
    loss_fn: LossFn,
    tx: optax.GradientTransformation,
):
    @jax.jit
    def train_step(
        state: TrainState,
        batch_x: Array,
        batch_y: Array,
    ) -> tuple[TrainState, dict[str, Array]]:
        key, dropout_key = jax.random.split(state.rng_key)


        def loss_with_params(params):
            model = nnx.merge(state.graphdef, params)
            model.train()
            rngs = nnx.Rngs(dropout=dropout_key)
            logits = model(batch_x, rngs=rngs) 
            return loss_fn(logits, batch_y)

        loss, grads = jax.value_and_grad(loss_with_params)(state.params)
        updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        new_state = TrainState(
            graphdef=state.graphdef,
            params=new_params,
            opt_state=new_opt_state,
            rng_key=key,
            step=state.step + jnp.array(1, dtype=jnp.int32),
        )

        metrics = {
            "train/loss": loss,
            "train/grad_norm": optax.global_norm(grads),
        }
        return new_state, metrics

    return train_step


def make_eval_step(eval_fn: EvalFn):
    @jax.jit
    def eval_step(
        state: TrainState,
        batch_x: Array,
        batch_y: Array,
    ) -> dict[str, Array]:
        model = nnx.merge(state.graphdef, state.params)
        model.eval()
        logits = model(batch_x) 
        return eval_fn(logits, batch_y)

    return eval_step


def run_eval(
    state: TrainState,
    loader_fn: Callable[[], DataLoader],
    eval_step: Callable[[TrainState, Array, Array], dict[str, Array]],
) -> dict[str, float]:
    totals: dict[str, Array] = {}
    total_examples = 0

    for batch_x, batch_y in loader_fn():
        metrics = eval_step(state, batch_x, batch_y)
        batch_size = batch_x.shape[0]
        total_examples += batch_size

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0) + v * batch_size

    if total_examples == 0:
        return {}

    return {
        f"eval/{k}": float(device_get(v)) / total_examples
        for k, v in totals.items()
    }

# ------------------
# Trainer
# ------------------

def train(
    model: nnx.Module,
    train_loader: DataLoader,
    loss_fn: LossFn,
    config: RuntimeConfig,
    *,
    n_train: int,
    eval_loader_fn: Callable[[], DataLoader] | None = None,
    eval_fn: EvalFn | None = None,
    checkpoint: CheckpointIO | None = None
) -> tuple[nnx.Module, TrainState]:
    if (eval_loader_fn is None) != (eval_fn is None):
        raise ValueError("Provide both eval_loader and eval_fn, or neither.")

    if config.drop_remainder:
        steps_per_epoch = max(1, n_train // config.batch_size)
    else:
        steps_per_epoch = max(1, (n_train + config.batch_size - 1) // config.batch_size)

    total_steps = steps_per_epoch * config.n_epochs

    tx, lr_schedule = build_tx(config, total_steps)
    state = build_train_state(model, tx, config.seed)
    train_step = make_train_step(loss_fn, tx)
    eval_step = make_eval_step(eval_fn) if eval_fn is not None else None

    step = int(device_get(state.step))

    wandb.init(project=config.project_name, config=config.model_dump())
    try:
        for batch_x, batch_y in train_loader:
            state, train_metrics = train_step(state, batch_x, batch_y)

            # step = int(device_get(state.step)) # kind of hate the constant get, but good to have in sync
            step += 1
            epoch = (step - 1) // steps_per_epoch

            if step % config.log_every == 0:
                log_data = {
                    "step": step,
                    "epoch": epoch,
                    "lr": float(device_get(lr_schedule(step - 1))),
                }
                logger.info("step = {}", step)
                for k, v in train_metrics.items():
                    log_data[k] = float(device_get(v))
                wandb.log(log_data)

            if (
                eval_loader_fn is not None
                and eval_step is not None
                and step % config.eval_every == 0
            ):
                wandb.log({
                    "step": step,
                    **run_eval(state, eval_loader_fn, eval_step),
                })

            if (
                checkpoint is not None
                and step % checkpoint.config.save_every == 0
            ):
                logger.debug("Save checkpoint to {}", checkpoint.config.dir)
                checkpoint.save(state, config)

        if checkpoint is not None:
            logger.debug("Save checkpoint to {}", checkpoint.config.dir)
            checkpoint.save(state, config)

        return materialize_model(state), state
    finally:
        wandb.finish()