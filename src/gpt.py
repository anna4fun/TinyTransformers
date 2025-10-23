import torch
from torch import nn
from torch.nn import functional as F
from config import DataConfig, ModelConfig

class GPTLanguageModel(nn.Module):
    def __init__(self, config:ModelConfig, data_config:DataConfig):
        super().__init__()
        # todo: how to customize head_size?
        self.config = config
        # token_embedding_table is an intermediate step from token embd to logits
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        # position_embedding_table contains semantic info about the position of the tokens
        self.position_embedding_table = nn.Embedding(data_config.block_size, config.n_embd)
        # a linear layer map embd to logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def token_position_embedding(self, idx, config:ModelConfig):
        B, T = idx.shape
        token_embedding_table = self.token_embedding_table(idx) # (B, T, n_embd)
        position_embedding_table = self.position_embedding_table(torch.arange(T)) # (T, n_embd)
        return token_embedding_table, position_embedding_table

    def forward(self, idx, targets=None):
        # idx is the index(numeric representation) of tokens passed to the model, there're B sentences, each sentence contains T tokens
        B, T = idx.shape
        token_embd, pos_embd = self.token_position_embedding(idx)
        x = token_embd + pos_embd # broadcast to (B, T, n_embd) + (T, n_embd) = (B, T, n_embd)
        logits = self.lm_head(x) # (B,T, vocab_size)
        if targets is not None:
            # logits.view(-1, logits.size(-1)):
            # logits.size(-1) means keeping the last dimension, the other -1 tells pytorch to infer the product of the first 2 dimensions
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None
        return logits, loss

