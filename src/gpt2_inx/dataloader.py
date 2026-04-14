from typing import Any
from collections.abc import Sequence

from grain import DataLoader, ReadOptions
from grain.samplers import IndexSampler
from grain.sharding import NoSharding
from grain.transforms import Batch

from jax import Array

from gpt2_inx.pipelines.data import XYSource
from gpt2_inx.configuration import RuntimeConfig

# ------------------
# Data loader
# ------------------


def make_source(data: tuple[Array, Array]) -> XYSource:
    return XYSource(data)


def make_sampler(
    *,
    num_records: int,
    seed: int,
    shuffle: bool,
    num_epochs: int,
) -> IndexSampler:
    return IndexSampler(
        num_records=num_records,
        num_epochs=num_epochs,
        shard_options=NoSharding(),
        shuffle=shuffle,
        seed=seed,
    )


def make_batch_operations(
    *,
    batch_size: int,
    drop_remainder: bool,
) -> list[Batch]:
    return [
        Batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder,
        )
    ]


def make_read_options(config: RuntimeConfig) -> ReadOptions:
    return ReadOptions(
        num_threads=config.n_threads,
        prefetch_buffer_size=config.prefetch_size,
    )


def make_loader(
    *,
    source: XYSource,
    sampler: IndexSampler,
    operations: Sequence[Any],
    config: RuntimeConfig,
) -> DataLoader:
    return DataLoader(
        data_source=source,
        sampler=sampler,
        operations=list(operations),
        worker_count=config.n_workers,
        worker_buffer_size=config.worker_buffer_size,
        read_options=make_read_options(config),
    )


def make_train_loader(
    data: tuple[Array, Array],
    config: RuntimeConfig,
) -> DataLoader:
    source = make_source(data)
    sampler = make_sampler(
        num_records=len(source),
        seed=config.seed,
        shuffle=True,
        num_epochs=config.n_epochs,
    )
    operations = make_batch_operations(
        batch_size=config.batch_size,
        drop_remainder=config.drop_remainder,
    )
    return make_loader(
        source=source,
        sampler=sampler,
        operations=operations,
        config=config,
    )


def make_eval_loader(
    data: tuple[Array, Array],
    config: RuntimeConfig,
) -> DataLoader:
    source = make_source(data)
    sampler = make_sampler(
        num_records=len(source),
        seed=config.seed,
        shuffle=False,
        num_epochs=1,
    )
    operations = make_batch_operations(
        batch_size=config.batch_size,
        drop_remainder=False,
    )
    return make_loader(
        source=source,
        sampler=sampler,
        operations=operations,
        config=config,
    )
