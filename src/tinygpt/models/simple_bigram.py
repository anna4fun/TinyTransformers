import torch
import torch.nn.functional as f
import torch.nn as nn
from tinygpt.configs.config import ModelConfig

class BigramLanguageModel(nn.Module):
    def __init__(self, config:ModelConfig):
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
        # Question: is idx = input tokens (e.g. "a", "b")  or the indexes of the input tokens (eg. [1,2])
        # Answer: idx = xb, targets = yb; xb, yb are (B,T) tensors of integers
        # idx is the input data with dimension (B,T), B=batch_size, T=time step (number of tokens in one line)
        B, T = idx.shape
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
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # idx starting dimension (B,T)
            # logits:
            logits, loss = self(idx) # equal to self.forward(idx) (B, T, C)
            # Get the last token of each line/sentence
            # idx = idx[:, -1,:] # wrong, idx is the index of input tokens, not the input tokens themselves
            logits = logits[:, -1, :] # (B, C)
            # get the probability distribution for the last token of each line/sentence
            prob = logits.softmax(dim=-1) # (B,C), (,C) is the current last token's probability distribution over the vocabulary
            # for every batch, sample one next token from the distribution
            idx_next = prob.multinomial(num_samples=1) # (B,1)
            # attach the generated next token to idx sequence
            idx = torch.cat([idx, idx_next], dim=1) # dim=1, merge on the column (B,T) + (B,1) = (B,T+1)
        return idx



