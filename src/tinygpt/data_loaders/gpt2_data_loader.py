import gc
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from datasets import tqdm, load_from_disk, load_dataset
from torch import Generator
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
import torch.distributed as dist
from tinygpt.configs.config import GPT2DataConfig
import tiktoken
import logging
import random
from tqdm import tqdm
from datasets import load_from_disk, Dataset as HFDataset  # Hugging Face Datasets
from tinygpt.logger.setup_logger import setup_logger

# ----- The GPT2 Data Loader ------
"""
Process:
1. Load the documents onto CPU
2. Tokenize documents with tiktoken, output is a long 1-D tensor
3. (Deprecated)Slice the long 1-D tensor into batches of X and Y without shuffling. Each X and Y are of size (B,T), each sequence of Y is one token left of each sequence of X.
3. Sampling B random starting points, and chunk B batches of tokens from the 1-D tensors.
4. Return a DataLoader containing x and y (both (B,T)) into the GPT language model 
Warning: do not move data to GPU yet, the entire corpus will be huge and it's better stay at CPU
"""

# -------------------------
# Logging
# -------------------------
# logger = logging.getLogger("lm_dataset")
# logger.setLevel(logging.DEBUG)          # Or INFO
# handler = logging.StreamHandler()
# formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)
#
# # prevent double logging if module imported multiple times
# logger.propagate = False
# Setup logger (critical: call after DDP rank is set)
# todo: move this to train_gpt2.py after DDP enabled dataloader is build and fully tested
cfg = GPT2DataConfig()
logger = setup_logger(cfg=cfg, train_name='shakespeare', local_rank=0)


# -------------------------
# Data Loader for single untokenized file
# -------------------------
@lru_cache(maxsize=1)
def load_corpus(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

@lru_cache(maxsize=1)
def load_tokens(cfg: GPT2DataConfig, text_path= None) -> Tuple[torch.Tensor, tiktoken.Encoding]:
    # 1. load the input text, the default input is the Tiny Shakespeare
    if text_path is None:
        corpus = load_corpus(cfg.shakes_text_path)
    else:
        corpus = load_corpus(text_path)

    # 2. tokenize the corpus with tiktoken
    encoder = tiktoken.get_encoding("gpt2")
    tokens = encoder.encode(corpus)  # a list
    return corpus, tokens, encoder

def train_valid_split(tokens: torch.Tensor, split_frac: float=0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(tokens)
    split = int(n * split_frac)
    train = tokens[:split]
    valid = tokens[split:]
    return train, valid

class LMSequenceDataset(Dataset):
    """
    Language-model dataset that returns a (x, y) tuple, where x, y are both (block_size, ) tensors
    from a 1D token tensor.
    Given starting index i, x is tokens[i:i+T], y is tokens[i+1:i+1+T].
    """

    # global counter for logging
    log_counter = 0
    max_log_calls = 1  # only log the first 1 calls to avoid spam

    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

        if len(self.tokens) < self.block_size + 1:
            raise ValueError("Token is not enough to build 1 block")
            # TODO: change to patch

    def __len__(self):
        # max starting index for generating full block sized X, Y pairs
        # max starting index range from (0, ..., len(token) - block_size - 1) inclusive
        return len(self.tokens) - self.block_size

    def __getitem__(self, start_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[start_idx:   start_idx+self.block_size]
        y = self.tokens[start_idx+1: start_idx+self.block_size+1]
        # -------- Logging Block ----------
        # Log only first few calls to avoid 10k lines of debug spam
        if LMSequenceDataset.log_counter < LMSequenceDataset.max_log_calls:
            logger.debug(
                f"[Dataset] __getitem__ call #{LMSequenceDataset.log_counter}\n"
                f"  start_idx      = {start_idx}\n"
                f"  x slice        = [{start_idx}:{start_idx + self.block_size}] "
                f"(shape: {tuple(x.shape)})\n"
                f"  y slice        = [{start_idx + 1}:{start_idx + self.block_size + 1}] "
                f"(shape: {tuple(y.shape)})\n"
                f"  x[:8] preview  = {x[:8].tolist()}\n"
                f"  y[:8] preview  = {y[:8].tolist()}"
            )
            LMSequenceDataset.log_counter += 1
        # ---------------------------------
        return x, y

# Wrap DataSet object containing pairs of 1-D x and y into a DataLoader
def make_dataloader(cfg: GPT2DataConfig,
                    text_path=None,
                    train_frac=0.9,
                    num_workers=1) -> Dict[str,DataLoader]:
    corpus, tokens, encoder = load_tokens(cfg, text_path)
    tokens = torch.LongTensor(tokens)
    train_tokens, valid_tokens = train_valid_split(tokens, train_frac)
    train_dataset = LMSequenceDataset(train_tokens, cfg.block_size)
    valid_dataset = LMSequenceDataset(valid_tokens, cfg.block_size)

    torch.manual_seed(cfg.seed)
    # Train DataLoader
    train_dl = DataLoader(train_dataset,
                          batch_size=cfg.batch_size,
                          num_workers=num_workers,
                          shuffle=cfg.shuffle,
                          pin_memory=True,
                          drop_last=False,)

    # Valid DataLoader
    valid_dl = DataLoader(valid_dataset,
                          batch_size=cfg.batch_size,
                          num_workers=num_workers,
                          shuffle=cfg.shuffle,
                          pin_memory=True,
                          drop_last=False,)
    return dict(train_dl=train_dl,valid_dl=valid_dl, encoder=encoder)


# -------------------------- Sharded LM Dataset (Resumable + DDP) --------------------------
class ResumableShardedLMSequenceDataset(Dataset):
    """
    P0:
    - Loads shards from pre-split train/test folders (no manual splitting)
    - Trackable progress for resumable training
    P1:
    - DDP-compatible (unique data per GPU)
    """

    def __init__(self, cfg: GPT2DataConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split  # "train" or "test" (uses pre-split folders)
        self.block_size = cfg.block_size
        # Step 1: List all shard paths (sorted) from pre-split folder
        self.shard_paths = self._get_sorted_shard_paths()
        # Optional: Pre-load a small buffer (avoids reloading files too often)
        self.current_file_idx = 0
        self.current_tokens = self._load_file(self.current_file_idx)
        self.current_pos = 0

    def _get_sorted_shard_paths(self) -> List[Path]:
        """Load shards from pre-split train/test folder (e.g., ./fineweb_edu/train/)"""
        split_dir = self.cfg.gpu_audodl_fineweb_path / self.split
        if not split_dir.exists:
            raise ValueError(f"Pre-split folder not found: {split_dir}")
        # List and sort shards numerically (shard_000.npy < shard_001.npy)
        shard_files = sorted(list(split_dir.glob("*.npy"))) # every element is a PosixPath object
        return shard_files

    def _load_file(self, file_idx):
        """Load a single .npy file into a numpy array of tokens"""
        file_path = self.shard_paths[file_idx]
        try:
            # Load tokens (returns 1D numpy array of integers)
            tokens = np.load(file_path).astype(np.int64)
            return tokens
        except Exception as e:
            print(f"Warning: Failed to load {file_path} – stopped. Error: {e}")

    def __len__(self):
        """
        We don't need an exact length – approximate is fine for training.
        This returns a large number (enough for epochs of training).
        """
        # Approx: 10B tokens / 1024 seq length = ~9.7M sequences (adjust as needed)
        return 10_000_000

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sequence of (seq_length + 1) tokens (for input/target)"""
        target_length = self.block_size + 1
        # Ensure all batches contains enough tokens
        while len(self.current_tokens) - self.current_pos < target_length:
            # keep all the tokens of the current shard
            remaining_tokens = self.current_tokens[self.current_pos:]
            # load tokens from next shard
            self.current_file_idx = (self.current_file_idx + 1) % len(self.shard_paths)
            new_tokens = self._load_file(self.current_file_idx)
            # stitch old and new shard's tokens together
            self.current_tokens = torch.cat([remaining_tokens, new_tokens], dim=0)
            self.current_pos = 0

        # Extract a sequence of (seq_length + 1) tokens
        # (GPT-2 uses input = tokens[:-1], target = tokens[1:])
        buf = self.current_tokens[self.current_pos: self.current_pos + self.block_size + 1]
        x = buf[:-1]  # inputs
        y = buf[1:]   # targets
        self.current_pos += self.block_size  # Move current_pos for next sequence
        return x, y


def make_ddp_dataloader(cfg: GPT2DataConfig, g: Generator) -> Dict[str, DataLoader]:
    """Build DDP-compatible DataLoaders with resumable training"""
    # Initialize DDP (call this once at the start of training)
    if cfg.ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")  # NCCL for GPU DDP

    # TODO: create validation set
    # Create train/test datasets (pre-split folders)
    full_train_dataset = ResumableShardedLMSequenceDataset(cfg, split="train")
    val_split_ratio = getattr(cfg, "val_split", 0.1)  # 10% val by default (add to your GPT2DataConfig)
    val_size = int(len(full_train_dataset) * val_split_ratio)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    test_dataset = ResumableShardedLMSequenceDataset(cfg, split="test")
    train_dl = DataLoader(train_dataset,
                          batch_size=cfg.batch_size,
                          # sampler=sampler,  # Overrides shuffle (use sampler for DDP)
                          num_workers=cfg.num_workers,
                          pin_memory=True,
                          drop_last=False,
                          persistent_workers=True if cfg.num_workers > 0 else False,
                          generator=g,  # locks the DataLoader's shuffle order
                          shuffle=True
                          )
    val_dl = DataLoader(val_dataset,
                          batch_size=cfg.batch_size,
                          # sampler=sampler,  # Overrides shuffle (use sampler for DDP)
                          num_workers=cfg.num_workers,
                          pin_memory=True,
                          drop_last=False, # Dropping samples leads to a biased validation score
                          persistent_workers=True if cfg.num_workers > 0 else False,
                          generator=g,
                          shuffle=True # reshuffle to avoid overfitting to the fixed subset of validation, unbiased estimator
                        )
    test_dl = DataLoader(test_dataset,
                          batch_size=cfg.batch_size,
                          # sampler=sampler,  # Overrides shuffle (use sampler for DDP)
                          num_workers=cfg.num_workers,
                          pin_memory=True,
                          drop_last=False,
                          persistent_workers=True if cfg.num_workers > 0 else False,
                          generator=g,
                          shuffle=False
                         )
    # DDP
    # train_dl = build_dl(train_dataset, "train")
    # test_dl = build_dl(test_dataset, "test")
    return dict(train_dl=train_dl, test_dl=test_dl, val_dl=val_dl)

# -------------------------- DDP DataLoader Builder --------------------------
# DDP: Use DistributedSampler (critical for batch splitting across GPUs)
# def build_dl(dataset: ResumableShardedLMSequenceDataset, split: str) -> DataLoader:
#     sampler = DistributedSampler(
#         dataset,
#         num_replicas=cfg.world_size,
#         rank=cfg.rank,
#         shuffle=cfg.shuffle,
#         seed=cfg.seed
#     ) if cfg.ddp else None
#
#     return DataLoader(
#         dataset,
#         batch_size=cfg.batch_size,
#         sampler=sampler,  # Overrides shuffle (use sampler for DDP)
#         num_workers=cfg.num_workers,
#         pin_memory=True,
#         drop_last=False,
#         persistent_workers=True if cfg.num_workers > 0 else False
#     )