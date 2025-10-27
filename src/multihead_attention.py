import torch
import torch.nn as nn
from config import ModelConfig, DataConfig
from self_attention_head import SelfAttentionHead

class MultiHeadAttention(torch.nn.Module):
    """ Multi-Head Attention module """
    """ We are going to distribute the overall attention training into multiple heads and then combine together"""
    def __init__(self, config: ModelConfig, data_config: DataConfig, num_heads):
        super().__init__()
        self.config = config
        self.data_config = data_config
        self.head_size = config.n_embd//num_heads # floor division
        # multi-head is a parallel partitioned list of Self-Attention heads(aka, sequence of aggregated token information: q @ k.T @ v)
        self.heads = nn.ModuleList([SelfAttentionHead(config, data_config, self.head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(self.head_size*num_heads, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        # combine all the self-attention heads (B,T,hs) into one big matrix (B,T, hs * num_heads)
        # each head(attention) represent different
        attention_combo = torch.cat([h(x) for h in self.heads], dim=-1)
        # project (B,T, hs * num_heads) into (B,T,C)
        attention_combo  = self.proj(attention_combo)
        out = self.dropout(attention_combo) #(B, T, C）
        return attention_combo, out