# train_gpt2.py
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

#----------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024  # T: time step, sequence length
    vocab_size: int = 50257
    n_embd: int = 768
    n_head: int = 12
    n_layers: int = 12 # see HF transformer parameters h.0 to h.11


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # naming convention follow the HuggingFace parameters
        self.c_fc = nn.Linear(config.n_embd, config.n_embd*4)
        # gelu: G for Gaussian, replace the 0 part of Relu to be waves of Tan
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd*4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # c_attn is a concatenation of Q,K,V, each is of dim (n_embd, n_embd)
        self.c_attn = nn.Linear(config.n_embd, config.n_embd*3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # bias is of dim (1, 1, T, T) (not (B, nh, T, T))
        self.register_buffer('bias',
                             torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size)
                             )


    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x) # (B, T, 3*n_embd)
        # Split `qkv` along dim=2(last dim), size of each chunk equals **n_embd**
        q, k, v = qkv.split(split_size_or_sections=self.n_embd, dim=2)
        # Transform the last dim C into a rectangular shape (nh, hs): nh: n_heads, hs: head size = n_embd//n_heads
        # transpose(1,2): swap dim 1 and 2 (from dim 0, 1, 2, 3)
        q = q.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        ## Attention
        # blob 1: attention weights
        attn = q @ k.transpose(2,3) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T), **1.0** not 1
        # broadcast the lower triangle mask [1,1,T,T] across [B, nh, T, T]
        mask = self.bias[0,0,T,T] == 0 # (1,1,T,T), True where to block, take only the “active” square
        attn = attn.masked_fill(mask, float('-inf'))  # (B, nh, T, T), replace the 0 with -inf
        attn = attn.softmax(dim=-1)  # (B, nh, T, T) # todo: verify dim, should be the last column
        # blob 2: contextual embedding
        y = attn @ v  # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        # swap the dimensions back to (B, T, nh, hs) and transform (nh, hs) back to (C)
        # calling .contiguous() makes a contiguous copy so view can safely collapse the last two dims.
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        # project the dimension back
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # Feed Forward

    def forward(self, x):
        x = x + self.multihead_attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            # ModuleDict allows you to index into submodules using keys
            dict(
                # nn.Embedding is a fancy wrapper of a tensor of values which we can access its element by indexing into the rows
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                # nn.ModuleList creates a model list so we can index it using integers like h.0 to h.11
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # the final layer norm added to the attention paper figure 2 after publishing
                ln_f = nn.LayerNorm(config.n_embd),
            )
        )
        # the classifier that project next word prediction from embedding vector into tokens in the vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
