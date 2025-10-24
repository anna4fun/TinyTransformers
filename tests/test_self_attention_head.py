import torch
from config import ModelConfig, DataConfig
from data_loader import make_dataloaders
from self_attention_head import SelfAttentionHead
from gpt import GPTLanguageModel

def test_self_attention():
    data_config = DataConfig(block_size=15, batch_size=8, val_frac=0.01, seed=42, shuffle=True)
    bundle = make_dataloaders(data_config)
    vocab_size = bundle["vocab_size"]
    model_config = ModelConfig(vocab_size=vocab_size, n_embd=10)
    # first load the GPT main model, we need it for the embedding
    model = GPTLanguageModel(model_config, data_config)
    xb, yb = next(iter(bundle["train_loader"]))
    te, pe = model.token_position_embedding(xb, model_config)
    x = te + pe # x is token + position embedded xb
    hs = 10
    att_init = SelfAttentionHead(model_config, data_config, head_size=hs)
    k,q, attention, scale = att_init.attention(x)
    print(scale)
    assert x.shape == (data_config.batch_size, data_config.block_size, model_config.n_embd)
    assert k.shape == (data_config.batch_size, data_config.block_size, hs)
    assert q.shape == (data_config.batch_size, data_config.block_size, hs)
    assert attention.shape == (data_config.batch_size, data_config.block_size, data_config.block_size)