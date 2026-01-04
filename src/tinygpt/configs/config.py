from dataclasses import dataclass
from typing import Optional

import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

# Here is the initial config for playground
# Data loading
# frozen=True: prevents accidental mutation
@dataclass(frozen=True)
class DataConfig:
    text_path: str = PROJECT_ROOT / "lecture_code_and_data/input.txt"
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
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.2

# Here is the config for HuggingFace GPT2 124M param model
@dataclass(frozen=True)
class GPT2DataConfig:
    batch_size: int = 64
    block_size: int = 1024
    shakes_text_path: str = PROJECT_ROOT / "lecture_code_and_data/input.txt"
    mps_fineweb_path: str = PROJECT_ROOT / "lecture_code_and_data/fineweb_edu_0048.npy"
    gpu_audodl_fineweb_path: Path = Path("/root/") / "autodl-tmp/kaggle-fineweb"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    checkpoint_every_n_shards: int = 2
    token_column: str = "tokens"
    num_workers: int = 4
    ddp: bool = False
    rank: int = 0
    world_size: int = 4
    resume_checkpoint: Optional[str] = None
    log_level: str = "DEBUG"  # DEBUG/INFO/WARNING/ERROR
    shuffle: bool = True
    seed: int = 1337
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    dropout: float = 0.2
    device: str = 'mps'
    eval_iters: int = 200
    vocab_size: int = 50257  # number of tokens: 50,000 BPE + 256 bytes tokens(the leaves of the BPE tree) + 1 <|endoftext|> token that delimites different documents
    n_embd: int = 768  # 768=64x12
    n_layer: int = 12
    n_head: int = 12

@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int = 16
    block_size: int = 32
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    shakes_text_path: str = PROJECT_ROOT / "lecture_code_and_data/input.txt"
    shuffle: bool = True
    seed: int = 1337
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    dropout: float = 0.2
    device: str = 'mps'
    eval_iters: int = 200
    vocab_size: int = 50257
