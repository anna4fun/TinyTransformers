import torch
import torch.nn.functional as f
import torch.nn as nn
from dataclasses import dataclass

# hyperparameters
@dataclass
class Config:
    vocab_size: int

class BigramLanguageModel(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        # The embedding table is the target and only output of the Bigram model
        # Embedding table is a lookup table for indexing,
        # each row correspond to 1 token ID,
        # each row vector is the logits(probability distribution) for the next token
        # aka, table[i, j] = p1 means: when current token is i, the probability of the next token being j is p1
        # sum(table[i,j], j = 1 to vocab_size) = 1
        self.embedding_table = nn.Embedding(config.vocab_size, config.vocab_size)

    def forward(self, idx, targets=None):
        # idx is the input data with dimension (B,T), B=batch_size, T=time step, number of tokens in one line
        logits = self.embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = f.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_token):
        for _ in range(max_new_token):
            logits, loss = self(idx)
            idx = idx[:, -1,:] #get the last token of each line/sentence
            # get the probability distribution out of the current embedding table


