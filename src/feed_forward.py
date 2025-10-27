import torch.nn as nn
from config import ModelConfig

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n_embd = config.n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        # todo: is the input x = output of the multihead attention? (bc L1's input dim = n_embd)
        return self.net(x)