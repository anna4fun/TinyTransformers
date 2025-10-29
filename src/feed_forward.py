import torch.nn as nn
from config import ModelConfig

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    # Feed Forward is an MLP, added right after the self attention and before the loss, it’s at the token level.
    # the self attention is the communication that first gather the affinity information
    # and then the FeedForward makes each token to think about those information independently.
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