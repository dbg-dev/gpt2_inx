from dataclasses import dataclass


@dataclass(slots=True)
class hyparams:
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    ff_hidden_dim: int
    ctx_len: int
    use_bias: bool = False
    dropout_rate: float = 0.1

@dataclass(slots=True)
class TrainerConfig:
    learning_rate: float
    num_epochs: int
    batch_size: int
    weight_decay: float
    train_ratio: float
    eval_freq: int
    eval_iter: int
