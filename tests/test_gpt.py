import torch
from tinygpt.models.gpt import GPTLanguageModel
from tinygpt.configs.config import DataConfig, ModelConfig
from tinygpt.data_loaders.data_loader import make_dataloaders

def test_token_position_embedding():
    data_config = DataConfig(block_size=15, batch_size=8, val_frac=0.01, seed=42, shuffle=True)
    bundle = make_dataloaders(data_config)
    vocab_size = bundle["vocab_size"]
    model_config = ModelConfig(vocab_size=vocab_size, n_embd=4)
    assert model_config.n_embd == 4
    model = GPTLanguageModel(model_config, data_config)
    xb, yb = next(iter(bundle["train_loader"]))
    idx = xb[0:2]
    B, T = idx.shape
    te, pe = model.token_position_embedding(idx) # no need to specify model_config
    assert te.shape  == (B, T, model_config.n_embd)
    assert pe.shape == (T, model_config.n_embd)
    # verify the broadcasting
    assert (te + pe).shape == (B, T, model_config.n_embd)
    # verify the reshape from (B,T,C) into (B*T,C)
    assert te.view(-1, te.size(-1)).shape == te.view(B*T, model_config.n_embd).shape
    assert torch.equal(pe.view(-1), pe.view(T*model_config.n_embd))