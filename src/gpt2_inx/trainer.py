from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, TypeAlias


import jax
import jax.numpy as jnp
import optax
import wandb

from grain import DataLoader, ReadOptions, MapDataset
from grain.samplers import IndexSampler
from grain.sharding import NoSharding#, ShardOptions
from grain.transforms import Batch as btch
# from grain.experimental import device_put as grain_device_put

from flax import struct
from flax import nnx
from jax import Array, device_get, device_put

from pathlib import Path
import orbax.checkpoint as ocp

from gpt2_inx.pipelines.data import XYSource


LossFn: TypeAlias = Callable[[Array, Array], Array]
EvalFn: TypeAlias = Callable[[Array, Array], dict[str, Array]]
Batch: TypeAlias = tuple[Array, Array]

@dataclass(slots=True)
class TrainerConfig:
    batch_size: int
    drop_remainder: bool
    n_epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    warmup_steps: int = 100
    min_learning_rate: float = 0.0
    seed: int = 0
    log_every: int = 100
    eval_every: int = 500
    prefetch_size: int = 2
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 1000
    n_workers: int = 0
    worker_buffer_size: int = 1


@struct.dataclass
class TrainState:
    graphdef: nnx.GraphDef[Any]
    params: nnx.State
    opt_state: optax.OptState
    rng_key: jax.Array
    step: Array  # jnp.int32 scalar


def make_loader(
    data: tuple[Array, Array],
    config: TrainerConfig,
    *,
    shuffle: bool,
):
    src = XYSource(data)

    sampler = IndexSampler(
        num_records=len(src),
        num_epochs=1, # only use the dataset for one epoch
        shard_options = NoSharding(),
        shuffle=shuffle,
        seed=config.seed,
    )

    operations = [
        btch(batch_size=config.batch_size, drop_remainder=config.drop_remainder)
    ]

    return DataLoader(
        data_source=src,
        sampler=sampler,
        operations=operations,
        worker_count=config.n_workers,
        worker_buffer_size = config.worker_buffer_size,
        read_options = ReadOptions(
            num_threads=0,          # recommended for in-memory data
            prefetch_buffer_size=0, # keep it simple at first
        ),
    )


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
    config: TrainerConfig,
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
    seed:int = 0
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
# Checkpointing
# ------------------

def _checkpoint_payload(state: TrainState, config: TrainerConfig) -> dict[str, Any]:
    return {
        "graphdef": state.graphdef,
        "params": state.params,
        "opt_state": state.opt_state,
        "rng_key": state.rng_key,
        "step": state.step,
        "config": asdict(config),
    }



def save_checkpoint(
    state: TrainState,
    config: TrainerConfig
) -> Path:
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = int(device_get(state.step))
    path = ckpt_dir / f"step_{step:08d}"

    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(path, _checkpoint_payload(state, config))

    return path


def restore_checkpoint(
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
        "config": {},
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
    loader: DataLoader,
    eval_step: Callable[[TrainState, Array, Array], dict[str, Array]]
) -> dict[str, float]:

    totals: dict[str, float] = {}
    count = 0

    for batch_x, batch_y in loader:
        metrics = eval_step(state, batch_x, batch_y)
        count += 1
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(device_get(v))

    if count == 0:
        return {}

    return {f"eval/{k}": v / count for k, v in totals.items()}



def train(
    model: nnx.Module,
    train_loader: DataLoader,
    loss_fn: LossFn,
    config: TrainerConfig,
    *,
    eval_loader: DataLoader | None = None,
    eval_fn: EvalFn | None = None,
) -> tuple[nnx.Module, TrainState]:
    n_train = len(train_loader._data_source)

    if config.drop_remainder:
        steps_per_epoch = max(1, n_train // config.batch_size)
    else:
        steps_per_epoch = max(1, (n_train + config.batch_size - 1) // config.batch_size)

    total_steps = steps_per_epoch * config.n_epochs

    tx, lr_schedule = build_tx(config, total_steps)
    state = build_train_state(model, tx)
    train_step = make_train_step(loss_fn, tx)
    eval_step = make_eval_step(eval_fn) if eval_fn is not None else None

    wandb.init(project="nnx-trainer", config=asdict(config))

    for _ in range(config.n_epochs):
        for batch_x, batch_y in train_loader:
            state, train_metrics = train_step(state, batch_x, batch_y)
            step = int(device_get(state.step))
            epoch = (step - 1) // steps_per_epoch

            if step % config.log_every == 0:
                log_data = {
                    "step": step,
                    "epoch": epoch,
                    "lr": float(device_get(lr_schedule(state.step - 1))),
                }
                for k, v in train_metrics.items():
                    log_data[k] = float(device_get(v))
                wandb.log(log_data)

            if (
                eval_loader is not None
                and eval_step is not None
                and step % config.eval_every == 0
            ):
                wandb.log({
                    "step": step,
                    **run_eval(
                        state,
                        eval_loader,
                        eval_step
                    ),
                })

        #     if config.save_every and step % config.save_every == 0:
        #         save_checkpoint(state, config, config.checkpoint_dir)

    # save_checkpoint(state, config)
    wandb.finish()
    return materialize_model(state), state