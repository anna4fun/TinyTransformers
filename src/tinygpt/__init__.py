# src/tinygpt/__init__.py

# Optional: Expose key modules/classes at the package level for easier imports
from tinygpt.models import simple_bigram, gpt, gpt2
from tinygpt.block import feed_forward, multihead_attention, self_attention_head, transformer_block
from tinygpt.data_loaders import data_loader, gpt2_data_loader
from tinygpt.local_tokenizers import character_tokenizers

# Optional: Define __version__ (matches pyproject.toml)
__version__ = "0.1.0"