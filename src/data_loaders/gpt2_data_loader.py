from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from config import GPT2DataConfig
import torch.nn as nn
import tiktoken
import logging


# ----- The GPT2 Data Loader ------
"""
Process:
1. Load the documents onto CPU
2. Tokenize documents with tiktoken, output is a long 1-D tensor
3. (Deprecated)Slice the long 1-D tensor into batches of X and Y without shuffling. Each X and Y are of size (B,T), each sequence of Y is one token left of each sequence of X.
3. Sampling B random starting points, and chunk B batches of tokens from the 1-D tensors.
4. Return a DataLoader containing x and y (both (B,T)) into the GPT language model 
"""

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("lm_dataset")
logger.setLevel(logging.DEBUG)          # Or INFO
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# prevent double logging if module imported multiple times
logger.propagate = False


# -------------------------
# IO
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
    Language-model dataset that returns (x, y) pairs of length block_size
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
