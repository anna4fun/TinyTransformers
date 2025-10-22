from dataclasses import dataclass
import torch

# Data loading
# frozen=True: prevents accidental mutation
@dataclass(frozen=True)
class DataConfig:
    text_path: str = "lecture_code_and_data/input.txt"
    block_size: int = 128
    batch_size: int = 64
    val_frac: float = 0.01
    seed: int = 1337
    num_workers: int = 0
    shuffle: bool = False
    drop_last: bool = False

# Model hyperparameters
# frozen=True: prevents accidental mutation
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 5
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    device: str = 'mps'
    eval_iters: int = 200
    n_embd: int = 32
    n_heads: int = 6
    n_layers: int = 6
    dropout: float = 0.2
