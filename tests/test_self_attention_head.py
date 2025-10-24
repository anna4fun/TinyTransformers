import torch
from config import ModelConfig, DataConfig
from data_loader import make_dataloaders
from self_attention_head import SelfAttentionHead
from gpt import GPTLanguageModel

def test_self_attention(self_attention_head_setup):
    s = self_attention_head_setup
    att_init = SelfAttentionHead(s.model_config, s.data_config, head_size=s.hs)
    k,q, attention, scale = att_init.attention(s.x)
    assert round(scale, 4) == 0.3162
    assert s.x.shape == (s.data_config.batch_size, s.data_config.block_size, s.model_config.n_embd)
    assert k.shape == (s.data_config.batch_size, s.data_config.block_size, s.hs)
    assert q.shape == (s.data_config.batch_size, s.data_config.block_size, s.hs)
    assert attention.shape == (s.data_config.batch_size, s.data_config.block_size, s.data_config.block_size)
    assert k.shape[-1] == s.hs

def test_self_attention_decoder(self_attention_head_setup):
    s = self_attention_head_setup
    att_init = SelfAttentionHead(s.model_config, s.data_config, head_size=s.hs)
    weight = att_init.decoder(s.x)
    # dimension should be (B,T,T)
    assert weight.shape == (s.data_config.batch_size, s.data_config.block_size,  s.data_config.block_size)
    # every batch is a TxT lower triangle, each row sum to 1, there should be Tx(T+1)/2 non-zero entries in each batch
    assert weight[0][3].sum().item() == 1
    assert torch.count_nonzero(weight[-1]) == s.data_config.block_size*(s.data_config.block_size+1)/2
    print(weight[0].sum(dim=1))

def test_self_attention_forward(self_attention_head_setup):
    s = self_attention_head_setup
    att_init = SelfAttentionHead(s.model_config, s.data_config, head_size=s.hs)
    output = att_init.forward(s.x)
    assert output.shape == (s.data_config.batch_size, s.data_config.block_size, s.hs)
    print(output[0])
