import argparse, torch
from pathlib import Path
from tinygpt.models.simple_bigram import BigramLanguageModel
from tinygpt.configs.config import ModelConfig
from tinygpt.local_tokenizers.character_tokenizers import build_tokenizers, decode  # your helpers

@torch.no_grad()
def sample(model, start_ids, max_new_tokens=200, temperature=1.0, top_k=None):
    model.eval()
    x = start_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :] / max(1e-6, temperature)  # (B, V)
        if top_k:
            v, i = torch.topk(logits, top_k)
            mask = logits < v[:, [-1]]
            logits[mask] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)      # (B, 1)
        x = torch.cat([x, next_id], dim=1)                     # (B, T+1)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt_simple_bigram.pt")
    ap.add_argument("--start", default="To be, or not to be")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    V = ckpt["vocab_size"]
    block_size = ckpt["block_size"]
    data_cfg = ckpt.get("cfg", {})
    text_path = data_cfg.get("text_path")
    if text_path is None:
        raise RuntimeError("Checkpoint missing text_path in cfg; cannot rebuild tokenizer.")

    # Rebuild tokenizer from the original corpus
    corpus = Path(text_path).read_text(encoding="utf-8")
    stoi, itos = build_tokenizers(corpus)

    # Rebuild model + load weights
    model = BigramLanguageModel(ModelConfig(vocab_size=V))
    model.load_state_dict(ckpt["model"])

    # Encode prompt and sample
    start_ids = torch.tensor([[stoi.get(ch, 0) for ch in args.start]], dtype=torch.long)
    out = sample(
        model,
        start_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=(args.top_k or None),
    )[0].tolist()

    print(decode(out, itos))

if __name__ == "__main__":
    main()
