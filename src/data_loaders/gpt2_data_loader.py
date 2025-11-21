from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import torch
from fastai.data.load import DataLoader
from torch.utils.data import Dataset

from config import GPT2DataConfig
import torch.nn as nn
import tiktoken


# ----- The GPT2 Data Loader ------
"""
Process:
1. Load the documents onto CPU
2. Tokenize documents with tiktoken, output is a long 1-D tensor
3. (Deprecated)Slice the long 1-D tensor into batches of X and Y without shuffling. Each X and Y are of size (B,T), each sequence of Y is one token left of each sequence of X.
3. Sampling B random starting points, and chunk B batches of tokens from the 1-D tensors.
4. Load X and Y into GPU
5. Return X as "idx" and Y as "target" into the GPT language model 
"""

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
    return corpus, tokens

def train_valid_split(tokens: torch.Tensor, split_frac: float=0.9) -> Dict[str, torch.Tensor]:
    n = len(tokens)
    split = int(n * split_frac)
    return {"train": tokens[:split], "valid": tokens[split:]}

class LMSequenceDataset(Dataset):
    """
    Language-model dataset that returns (x, y) pairs of length block_size
    from a 1D token tensor. x is tokens[i:i+T], y is tokens[i+1:i+1+T].
    """
    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

        if len(self.tokens) < self.block_size + 1:
            raise ValueError("Token is not enough to build 1 block")

    def __len__(self):
        # max starting index for generating full block sized X, Y pairs
        return len(self.tokens) - self.block_size - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        T = self.block_size
        x = self.tokens[idx:idx+T]
        y = self.tokens[idx+1:idx+T+1]
        return x, y


def make_dataloader(cfg: GPT2DataConfig,
                    text_path=None,
                    train_frac=0.9,
                    num_workers=0) -> Dict[str,DataLoader]:
    corpus, tokens = load_tokens(cfg, text_path)
    tokens = torch.LongTensor(tokens)
    train_dataset = LMSequenceDataset(tokens, cfg.shakes_text_path)
