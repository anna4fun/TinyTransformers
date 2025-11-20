import math
import time

from torch.optim import AdamW
from data_loaders.data_loader import *
from config import ModelConfig, DataConfig
from simple_bigram import BigramLanguageModel

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total, denom = 0.0, 0
    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits, loss = model(idx=xb, targets=yb)
        bs = xb.size(0)
        total += loss.item() * bs
        denom += bs
    return total / max(1, denom)


def main():
    # ---------------- Config ----------------
    cfg = DataConfig(
            block_size=20, batch_size=64, val_frac=0.01,
            seed=42, num_workers=0, shuffle=True, drop_last=True
        )
    if torch.cuda.is_available():
        device = "gpu"
    elif torch.backends.mps.is_available():
        device =  "mps"
    else:
        device = "cpu"
    torch.manual_seed(cfg.seed)

    # ---------------- Data ----------------
    bundle = make_dataloaders(cfg)
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    C = bundle["vocab_size"] # vocab_size = n_embds here
    T = bundle["block_size"]
    model_config = ModelConfig(vocab_size=C, learning_rate=1e-2) # the default lr=3e-5 which is too small

    # ---------------- Model ----------------
    model = BigramLanguageModel(model_config).to(device)
    # print(f"Model: BigramLM  | params: {count_params(model) / 1e6:.3f}M  | V={V}  | block={T}")

    # ---------------- Optim ----------------
    opt = AdamW(model.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay)

    # ---------------- Train ----------------
    steps = 3000
    log_every = 300
    best_val = float("inf")

    t0 = time.time()
    model.train()
    it = 0
    while it < steps:
        for xb, yb in train_loader:
            xb = xb.to(device)  # (B,T)
            yb = yb.to(device)  # (B,T)
            logits, loss = model.forward(idx=xb,targets=yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            it += 1
            if it % log_every == 0 or it == 1:
                val_loss = evaluate(model, val_loader, device)
                ppl = math.exp(val_loss) if val_loss < 20 else float("inf") # todo: what's ppl?
                print(f"it={it:5d}/{steps}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}  ppl={ppl:.2f}")

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        "model": model.state_dict(),
                        "vocab_size": C,
                        "block_size": T,
                        "cfg": cfg.__dict__,
                    }, "checkpoints/ckpt_simple_bigram.pt")

            if it >= steps:
                break

    print(f"Done in {time.time() - t0:.1f}s. Best val_loss={best_val:.4f}")

if __name__ == "__main__":
    main()