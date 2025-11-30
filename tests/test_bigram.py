from tinygpt.models.simple_bigram import *
from tinygpt.configs.config import *
import torch


# my original tests before conftest creation
def test_embedding_table_shape() -> BigramLanguageModel:
    torch.manual_seed(42)
    model = BigramLanguageModel(ModelConfig(vocab_size=5)) # C = 5
    test_idx = torch.tensor([[1,2,3]]) # (B=1, T=3)
    target_idx = torch.tensor([[1,2,3]])
    logits, loss = model.forward(idx = test_idx, targets = target_idx)
    print(logits)
    print(target_idx)
    print(logits.softmax(dim=-1).sum(dim=1))
    print(loss)
    assert logits.shape == (3,5) # logits collapsed (B, T, C) to (B*T, C)

# the following tests are added after conftest creation
def test_forward_shapes(bigram, vocab_size):
    x = torch.tensor([[0, 1, 2], [3, 4, 1]], dtype=torch.long)
    logits, loss = bigram(x)
    assert logits.shape == (2, 3, vocab_size)
    assert loss is None

def test_training_step_loss(bigram):
    x = torch.tensor([[0,1],[2,3]], dtype=torch.long)
    y = torch.tensor([[1,2],[3,4]], dtype=torch.long)
    _, loss = bigram(x, y)
    assert loss is not None
    assert loss.ndim == 0 and torch.isfinite(loss)

def test_generate_appends_tokens(bigram):
    x = torch.tensor([[0, 1]], dtype=torch.long)   # (B=1, T=2)
    out = bigram.generate(x, max_new_tokens=3)
    assert out.shape == (1, 5)
