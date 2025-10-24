import torch
from torch import nn
from torch.nn import functional as F
from config import DataConfig, ModelConfig

class SelfAttentionHead(nn.Module):
    def __init__(self, config: ModelConfig, data_config: DataConfig, head_size: int):
        super().__init__()
        self.config = config
        self.data_config = data_config
        self.head_size = head_size
        # k, q, v should all match with the number of tokens(block_size/time step)
        # (correct) head_size is n_embd/num of heads
        ## The following line of comment is wrong
        # "the 3 block_size x head_size matrix is the training targets of self attention"
        # k, q, v are not training targets, they are learned projection of the inputs (x), which are used internally to compute the attention (for each token, which other tokens should I learn information)
        # the training targets on the other hand are: 1. next word prediction 2. language translation
        # the self attention itself doesn't have its own target, it just learns to transform representations in a way that minimizes the model’s overall loss.
        ## The following 3 lines are wrong: k, v, q are size (B, T, head_size), k = x @ Wk = (B, T, n_embd) x (T, head_size), same as v and q
        # self.k = nn.Linear(data_config.block_size, head_size)
        # self.v = nn.Linear(data_config.block_size, head_size)
        # self.q = nn.Linear(data_config.block_size, head_size)
        # Encodes how each token can be matched by others
        self.Wk = nn.Linear(in_features = config.n_embd, out_features = head_size, bias=False) # Wk.weight is the learnable parameter , the matrix (head_size, n_embd) (notice it's transposed)
        # Encodes what information each token is looking for
        self.Wq = nn.Linear(in_features = config.n_embd, out_features = head_size, bias=False)
        # Encodes what information can each token offers once it the kxq found things that catches its attention (private information)
        self.Wv = nn.Linear(in_features = config.n_embd, out_features = head_size, bias=False)
        # These 3 weights are shared across all tokens and batches — learned during training
        # The Transformer’s power comes from the fact that the same Wq, Wk, Wv are applied to every token, allowing generalization to variable-length sequences.
        self.register_buffer('tril', torch.tril(torch.ones(data_config.block_size, data_config.block_size))) # (T,T) lower triangle

    def attention(self, x):
        # x should be (B,T, n_embd), which includes token embedding and positional embedding
        k = self.Wk(x)  # k = x @ Wk, (B,T, hs) ; hs = head_size
        q = self.Wq(x)  # q = x @ Wq, (B,T, hs)
        attention = q @ k.transpose(-2, -1) # * k.shape[-1] ** -0.5  # (B,T,T)
        scale_param = k.shape[-1] ** -0.5
        return k, q, attention, scale_param

    def forward(self, x):
        # x should be (B,T, n_embd)
        v = self.Wv(x)  # v = x @Wv,  (B,T, hs)
        k,q, attention, scale = self.attention(x)
        attention = attention * scale


