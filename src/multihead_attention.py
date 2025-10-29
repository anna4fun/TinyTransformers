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
        # Equally slice the entire embedding space into **num_heads** number of sub embedding spaces
        assert config.n_embd % num_heads == 0, "n_embd must be divisible by num_heads"
        self.head_size = config.n_embd//num_heads # floor division
        # Multi-head is a list of Self-Attention heads(aka, contextual embedding produced by: q @ k.T @ v)
        self.heads = nn.ModuleList([SelfAttentionHead(config, data_config, self.head_size) for _ in range(num_heads)])
        # Projection head that align the dimension of concatenated heads into the same dimension as the original embedding space
        self.proj = nn.Linear(self.head_size*num_heads, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        """ Combine all the self-attention heads (B,T,hs) into one big matrix (B,T, hs * num_heads) """
        # x is (B,T,C), C = n_embd
        # Each head learns different contextual embeddings: lexical identity (“cat”, “dog”), part of speech, semantic role, syntactic dependencies, discourse context
        # TODO: try examine the interpretability of each head
        # h(x) would call SelfAttentionHead.forward(x);  In PyTorch, when you call a module like a function (h(x)), it automatically invokes its .forward() method. And also handles devices and train/eval modes.
        # each h() is an object of type SelfAttentionHead, it has input dim: (B,T,C) and output dim: (B,T, hs)
        #    -> interpretation: every single head read the whole input x, and learn its own head_size contextual embedding
        attention_combo = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,hs) + (B,T,hs) =  (B,T, hs * num_heads)
        # project (B,T, hs * num_heads) into (B,T,C)
        attention_combo  = self.proj(attention_combo)
        out = self.dropout(attention_combo) #(B, T, C）
        return out  #(B, T, C）