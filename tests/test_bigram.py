from simple_bigram import *
import torch

def test_config():
    config = Config(5)
    assert config.vocab_size == 5

def test_embedding_table_shape():
    torch.manual_seed(42)
    model = BigramLanguageModel(Config(vocab_size=5)) # C = 5
    test_idx = torch.tensor([[1,2,3]]) # (B=1, T=3)
    target_idx = torch.tensor([[1,2,3]])
    logits, loss = model.forward(idx = test_idx, targets = target_idx)
    print(logits)
    print(target_idx)
    print(logits.softmax(dim=-1).sum(dim=1))
    print(loss)
    assert logits.shape == (3,5) # logits collapsed (B, T, C) to (B*T, C)