import torch
from torch import nn
from torch.nn import functional as F
from config import DataConfig, ModelConfig
from transformer_block import TransformerBlock
from feed_forward import FeedForward

class GPTLanguageModel(nn.Module):
    def __init__(self, config:ModelConfig, data_config:DataConfig):
        super().__init__()
        # todo: how to customize head_size?
        self.config = config
        # token_embedding_table is an intermediate step from token embd to logits
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        # position_embedding_table contains semantic info about the position of the tokens
        self.position_embedding_table = nn.Embedding(data_config.block_size, config.n_embd)
        # transformer block
        self.block = nn.Sequential(*[TransformerBlock(config, data_config, num_heads = config.n_heads) for _ in range(config.n_layers)])
        # final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        # a linear layer map embd to logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def token_position_embedding(self, idx):
        B, T = idx.shape
        token_embedding_table = self.token_embedding_table(idx) # (B, T, n_embd)
        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        position_embedding_table = self.position_embedding_table(pos) # (T, n_embd)
        return token_embedding_table, position_embedding_table

    def forward(self, idx, targets=None):
        # idx is the index(numeric representation) of tokens passed to the model, there are B sentences, each sentence contains T tokens
        B, T = idx.shape
        token_embd, pos_embd = self.token_position_embedding(idx)
        # TODO: would position embedding and token embedding be trained separately?
        x = token_embd + pos_embd # broadcast to (B, T, n_embd) + (T, n_embd) = (B, T, n_embd)
        # Transformer Block
        x = self.block(x) # in (B,T,C) todo: out dimension
        # Layer Norm
        x = self.ln_f(x) # in (B,T,C)
        # Feed forward is an MLP, added right after the self attention and before the loss, it’s at the token level -> the self attention is the communication that first gather the affinity information and then the FF makes each token to think about those information independently .
        logits = self.lm_head(x) # (B,T, vocab_size)
        # Question: would the multi-head attention be added here? No, logits is the last step before comparing with the targets and evaluate loss. everything else happens before logits.
        if targets is not None:
            # logits.view(-1, logits.size(-1)):
            # logits.size(-1) means keeping the last dimension, the other -1 tells pytorch to infer the product of the first 2 dimensions
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # todo: where is the backprop happening to updates the weights with gradients?
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens, config):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

