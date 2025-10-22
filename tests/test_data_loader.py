from config import DataConfig
from data_loader import make_dataloaders, decode
import torch

def test_make_dataloaders():
    cfg = DataConfig(block_size=15, batch_size=8, val_frac=0.01, seed=42, shuffle=True)
    torch.manual_seed(cfg.seed)
    bundle = make_dataloaders(cfg)
    assert bundle["vocab_size"] == 65
    assert bundle["train_loader"].batch_size == 8
    xb, yb = next(iter(bundle["train_loader"]))
    assert xb.shape == (8,15)
    assert yb.shape ==  (8,15)
    assert torch.equal(xb[0], torch.tensor([54, 56, 47, 60, 39, 58, 43,  1, 53, 56, 42, 43, 56,  1, 43]))
    # verify y is one character right shift of x
    assert decode(xb[5].tolist(), bundle["itos"]) == "her best array "
    assert decode(yb[5].tolist(), bundle["itos"]) == "er best array b"
