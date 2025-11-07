# train_gpt2.py
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

#----------------------------------------------
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
        """ Attention Weights and Contextual Embeddings """
        # GPT-2 (124M) have n_head = 12 and head_size = 64
        # so there are 12*64 = 768 channels in the Transformer
        B, T, C = x.shape
        qkv = self.c_attn(x) # (B, T, 3*n_embd)
        # Split `qkv` along dim=2(last dim), size of each chunk equals **n_embd**
        q, k, v = qkv.split(split_size_or_sections=self.n_embd, dim=2)
        # Transform the last dim C into a rectangular shape (nh, hs): nh: n_heads, hs: head size = n_embd//n_heads
        # transpose(1,2): swap dim 1 and 2 (from dim 0, 1, 2, 3)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        ## Attention
        # blob 1: attention weights
        attn = q @ k.transpose(2,3) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T), **1.0** not 1
        # broadcast the lower triangle mask [1,1,T,T] across [B, nh, T, T]
        mask = self.bias[0,0,T,T] == 0 # (1,1,T,T), True where to block, take only the “active” square
        attn = attn.masked_fill(mask, float('-inf'))  # (B, nh, T, T), replace the 0 with -inf
        attn = attn.softmax(dim=-1)  # (B, nh, T, T), softmax the last column
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


@dataclass
class GPTConfig:
    block_size: int = 1024  # T: time step, sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE + 256 bytes tokens(the leaves of the BPE tree) + 1 <|endoftext|> token that delimites different documents
    n_embd: int = 768 # n_head = 12 * head_size = 64 = 768
    n_head: int = 12
    n_layer: int = 12 # see HF transformer parameters h.0 to h.11


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

    # port the params from HuggingFace gpt2 and use it to initialize our model
    # the following module is a constructor that returns the GPT object if we just give it the model type (one of the 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict() # the state dict for HF model and our own model
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this autoregressive mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'] # in openai in TensorFlow have some of these tensors transposed vs. the PyTorch version, so we manually transpose them back.
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# --------------------------------------------------------------------------------------------
model = GPT.from_pretrained('gpt2')
print("didn't crash yay")