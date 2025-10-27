from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple, Dict, List, TypedDict
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from config import DataConfig

# -------------------------
# Config
# -------------------------
class DataBundle(TypedDict):
    stoi: Dict[str, int]
    itos: Dict[int, str]
    train_loader: DataLoader
    val_loader: DataLoader
    vocab_size: int
    block_size: int

# -------------------------
# Tokenization
# -------------------------
# The simplest tokenizer, every tokenizer contains the rule of how to break words and sentences, and a pair of encode and decode
def build_tokenizers(text: str) -> Tuple[Dict[str,int], Dict[int,str]]:
    # NOTE: char-level; includes whitespace and punctuation by design
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(s: str, stoi: Dict[str,int]) -> List[int]:
    # Will KeyError on unseen chars; fine as long as you fit on the full corpus.
    return [stoi[ch] for ch in s]

def decode(ids: List[int], itos: Dict[int,str]) -> str:
    return ''.join(itos[i] for i in ids)

# -------------------------
# IO
# -------------------------
@lru_cache(maxsize=1)
def load_corpus(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

# -------------------------
# Dataset
# -------------------------
class BlockDataset(Dataset):
    """Character-level next-token dataset producing (x, y) blocks."""
    def __init__(self, ids: Tensor, block_size: int):
        if ids.ndim != 1:
            raise ValueError("ids must be a 1D tensor of token ids")
        if len(ids) <= block_size:
            raise ValueError(
                f"Not enough tokens ({len(ids)}) for block_size={block_size}. "
                "Increase corpus or reduce block_size."
            )
        self.ids = ids
        self.block = block_size

    def __len__(self) -> int:
        return len(self.ids) - self.block

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        x = self.ids[i : i + self.block]          # (T,)
        y = self.ids[i + 1 : i + 1 + self.block]  # (T,)
        return x, y

# -------------------------
# Dataloaders
# -------------------------
def make_dataloaders(cfg: DataConfig) -> DataBundle:
    text = load_corpus(cfg.text_path)
    if not text:
        raise ValueError(f"Empty corpus at {cfg.text_path}")

    stoi, itos = build_tokenizers(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    # Hold out at least one full block for validation
    min_val = cfg.block_size + 1
    n_val = max(min_val, int(len(data) * cfg.val_frac))
    n_val = min(n_val, max(0, len(data) - (cfg.block_size + 1)))  # keep train ≥ one block
    if n_val == 0:
        raise ValueError(
            "Validation split would be empty. Increase corpus, increase val_frac, "
            "or reduce block_size."
        )

    train_ids = data[:-n_val]
    val_ids   = data[-n_val:]

    train_ds = BlockDataset(train_ids, cfg.block_size)
    val_ds   = BlockDataset(val_ids,   cfg.block_size)

    has_cuda = torch.cuda.is_available()
    pin = has_cuda  # only pin if CUDA exists

    # Only pass generator when shuffling (deterministic)
    generator = torch.Generator().manual_seed(cfg.seed) if cfg.shuffle else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=cfg.drop_last,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=False,  # usually keep all val batches
    )

    return {
        "stoi": stoi,
        "itos": itos,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "vocab_size": len(stoi),
        "block_size": cfg.block_size,
    }
