from functools import lru_cache
from pathlib import Path
from typing import Dict

import torch
from fastai.data.load import DataLoader

from config import GPT2DataConfig
import torch.nn as nn
import tiktoken


# ----- The GPT2 Data Loader ------
"""
Process:
1. Load the documents onto CPU
2. Tokenize documents with tiktoken, output is a long 1-D tensor
3. Slice the long 1-D tensor into batches of X and Y without shuffling. Each X and Y are of size (B,T), each sequence of Y is one token left of each sequence of X.
4. Load X and Y into GPU
5. Return X as "idx" and Y as "target" into the GPT language model 
"""

# -------------------------
# IO
# -------------------------
@lru_cache(maxsize=1)
def load_corpus(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def make_dataloader(cfg: GPT2DataConfig, text_path=None) -> Dict:
    # 1. load the input text, the default input is the Tiny Shakespeare
    if text_path is None:
        corpus = load_corpus(cfg.shakes_text_path)
    else:
        corpus = load_corpus(text_path)

    # 2. tokenize the corpus with tiktoken
    encoder = tiktoken.get_encoding("gpt2")
    tokens = encoder.encode(corpus) # a list
    tokens = torch.LongTensor(tokens)

    # 3. Batch the tokens
    num_batch = -(-len(tokens)//(cfg.batch_size * cfg.block_size))
    x = tokens.view(num_batch, cfg.batch_size, cfg.block_size)
    y = tokens[1:].view(num_batch, cfg.batch_size, cfg.block_size)

    return {
        "input_text": corpus,
        "tokens": tokens,
        "idx": x,
        "target": y,
    }