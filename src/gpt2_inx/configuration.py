from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class RuntimeConfig(BaseModel):
    project_name: str = Field(exclude=True)

    batch_size: int = Field(gt=0)
    drop_remainder: bool

    n_epochs: int = Field(default=1, ge=1)

    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    grad_clip_norm: float = Field(default=1.0, gt=0.0)

    warmup_steps: int = Field(default=100, ge=0)
    min_learning_rate: float = Field(default=0.0, ge=0.0)

    seed: int = 42

    log_every: int = Field(default=100, ge=1)
    eval_every: int = Field(default=500, ge=1)

    prefetch_size: int = Field(default=0, ge=0)
    n_workers: int = Field(default=0, ge=0)
    n_threads: int = Field(default=0, ge=0)
    worker_buffer_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_learning_rates(self) -> "RuntimeConfig":
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be <= learning_rate")
        return self

    model_config = { "extra": "forbid", }


class CheckpointConfig(BaseModel):
    dir: Path = Field(default=Path("./checkpoints"), validate_default=True)
    save_every: int = Field(default=1000, ge=1)

    @field_validator("dir")
    @classmethod
    def make_absolute(cls, v: Path) -> Path:
        return v.expanduser().resolve()
        
    model_config = { "extra": "forbid", }

