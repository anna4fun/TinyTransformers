from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader, random_split

@dataclass
class DataConfig:
    text_path: str
    block_size: int = 128
    batch_size: int = 32
    num_workers: int = 0  # set >0 for big corpora
    val_frac: float = 0.01
    seed: int = 0

def build_tokenizers(text: str) -> Tuple[Dict[str,int], Dict[int,str]]:
    chars = sorted(list(set(text)))
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for ch, i in stoi.items()}
    return stoi, itos

def encode(s: str, stoi: Dict[str,int]) -> List[int]:
    return [stoi[ch] for ch in s]

def decode(ids: List[int], itos: Dict[int,str]) -> str:
    return ''.join(itos[i] for i in ids)

@lru_cache(maxsize=1)
def load_corpus(path: str) -> str:
    # Cached so multiple callers don’t re-read disk
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

class BlockDataset(Dataset):
    """Character-level next-token dataset producing (x, y) blocks."""
    def __init__(self, ids: torch.Tensor, block_size: int):
        self.ids = ids
        self.block = block_size

    def __len__(self) -> int:
        return len(self.ids) - self.block

    def __getitem__(self, i: int):
        x = self.ids[i:i+self.block]           # (T,)
        y = self.ids[i+1:i+self.block+1]       # (T,)
        return x, y

def make_dataloaders(cfg: DataConfig):
    text = load_corpus(cfg.text_path)
    stoi, itos = build_tokenizers(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    # Simple train/val split on token level (for demo)
    n_val = int(len(data) * cfg.val_frac)
    train_ids, val_ids = data[:-n_val], data[-n_val:]

    train_ds = BlockDataset(train_ids, cfg.block_size)
    val_ds   = BlockDataset(val_ids,   cfg.block_size)

    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    vocab_size = len(stoi)
    return {
        "stoi": stoi, "itos": itos,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "vocab_size": vocab_size,
    }
