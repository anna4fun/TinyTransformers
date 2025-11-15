from torch import nn
from config import ModelConfig, DataConfig
from feed_forward import FeedForward
from multihead_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, config: ModelConfig, data_config: DataConfig, num_heads):
        # num_heads: the number of heads we'd like
        # layer norm: normalize every sequence
        super().__init__()
        self.config = config
        self.data_config = data_config
        # pre-norm attention
        # TODO: try other normalization, RMSNorm
        self.layer_norm1 = nn.LayerNorm(normalized_shape=config.n_embd)
        self.attention = MultiHeadAttention(config=config, data_config=data_config, num_heads=num_heads) # in (B,T,C), out (B,T,C)
        # pre-norm MLP feed forward
        self.layer_norm2 = nn.LayerNorm(normalized_shape=config.n_embd)
        self.ffwd = FeedForward(config=config)

    def forward(self, x):
        # x is shape (B,T,C), C = n_embd
        # pre-norm attention: layer norm first, then attention; this is different from the post-norm paradigm of the Attention paper Figure 2.
        # x = x + contextual embedding of x -> residual of x
        # this way we keep a clean residual path way from targets all the way back to the input text
        # the gradients also equally split between the x and the Attention+MLP blocks
        x = self.layer_norm1(x)
        x = x + self.attention(x) # attention(x): MultiHeadAttention.forward(x)
        # pre-norm feed forward
        x = self.layer_norm2(x)
        x = x + self.ffwd(x)
        return x