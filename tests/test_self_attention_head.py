import pytest
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
    weight = att_init.calc_weights(s.x, decoder=True)
    # dimension should be (B,T,T)
    assert weight.shape == (s.data_config.batch_size, s.data_config.block_size,  s.data_config.block_size)
    # every batch is a TxT lower triangle, each row sum to 1, there should be Tx(T+1)/2 non-zero entries in each batch
    assert torch.count_nonzero(weight[-1]) == s.data_config.block_size*(s.data_config.block_size+1)/2
    # each row sum to 1, so the sum of weights should equal to T
    # test with tolerance style 1: avoid error caused by float32: 14.999999 vs 15.
    assert weight[0].sum().item() == pytest.approx(s.data_config.block_size, rel=0, abs=1e-6)

def test_self_attention_encoder(self_attention_head_setup):
    s = self_attention_head_setup
    att_init = SelfAttentionHead(s.model_config, s.data_config, head_size=s.hs)
    weight = att_init.calc_weights(s.x, decoder=False)
    # dimension should be (B,T,T)
    assert weight.shape == (s.data_config.batch_size, s.data_config.block_size,  s.data_config.block_size)
    # each row sum to 1, so the sum of weights should equal to T
    # test with tolerance style 2: avoid error caused by float32: 14.999999 vs 15.
    assert torch.allclose(weight[0].sum(dim=1), torch.ones_like(weight[0].sum(dim=1)), atol=1e-6)

def test_self_attention_forward(self_attention_head_setup):
    s = self_attention_head_setup
    att_init = SelfAttentionHead(s.model_config, s.data_config, head_size=s.hs)
    output = att_init.forward(s.x)
    assert output.shape == (s.data_config.batch_size, s.data_config.block_size, s.hs)
    print(output[0])
    # The first row of the output[0] are all zero after adding the dropout, the values of each row also changed, investigate why
    # 1. “Whole first row became zeros” is expected with causal masks + attn-dropout. For the very first token, the causal mask may allow only one valid key (itself). If dropout removes that single prob, the row becomes all zeros.
    # 2. It’s normal that kept entries change value. PyTorch uses inverted dropout: kept values are scaled by 1/(1-p). So you won’t see “unchanged” 70%—they’re scaled up.
    """
    No dropout
    [100%]tensor([[ 0.8580,  0.4630, -1.0314, -0.8062, -1.5896,  0.8725,  0.2528,  1.1281,
          1.2933, -0.4617],
        [ 0.5380, -0.1424, -0.5944, -0.3881, -0.9990,  0.6399,  0.1829,  0.5744,
          0.9477, -0.1930],
        [ 0.1772, -0.1542, -0.0793, -0.5813, -0.4481, -0.0464,  0.3235, -0.1872,
          0.5260,  0.0583],
          
    With dropout
    PASSED    [100%]tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [-0.3963, -0.3139,  0.8846,  0.3196,  0.1869, -0.5924, -0.2424, -0.5869,
          0.0338,  0.1198],
        [-0.6518, -0.3664,  1.4674,  0.6425,  0.9364, -0.8745, -0.6350, -0.3583,
         -0.2629,  0.4405],
    """
