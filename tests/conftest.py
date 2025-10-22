# tests/conftest.py
from __future__ import annotations
import os
import random
import torch
import pytest

# --- Global settings ---------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def set_env_for_tests():
    # Keep tests deterministic & quiet
    os.environ.setdefault("PYTHONHASHSEED", "0")

@pytest.fixture(autouse=True)
def set_random_seeds():
    random.seed(1337)
    torch.manual_seed(1337)

# --- Devices ----------------------------------------------------------------

@pytest.fixture(scope="session")
def device():
    # Keep CPU by default for speed & consistency; flip to CUDA if you want.
    return torch.device("cpu")

# --- Configs ----------------------------------------------------------------

@pytest.fixture(scope="session")
def vocab_size() -> int:
    return 65

@pytest.fixture(scope="session")
def bigram_config(vocab_size):
    # Your dataclass config; import from your package
    from config import ModelConfig
    return ModelConfig(vocab_size=vocab_size)

# --- Model factories ---------------------------------------------------------

@pytest.fixture
def bigram(bigram_config):
    from simple_bigram import BigramLanguageModel
    return BigramLanguageModel(bigram_config)

# Example: a tiny transformer with small dims so unit tests are fast.
# @pytest.fixture
# def tiny_transformer(vocab_size):
#     from nlp_models.transformer import TransformerLM, TransformerConfig
#     cfg = TransformerConfig(
#         vocab_size=vocab_size,
#         d_model=64,
#         n_heads=2,
#         n_layers=2,
#         ff_mult=2,
#         block_size=32,
#         dropout=0.0,
#     )
#     return TransformerLM(cfg)

# --- IO helpers --------------------------------------------------------------

@pytest.fixture
def toy_tokens(vocab_size):
    # Small, deterministic token batch
    return torch.randint(0, vocab_size, (2, 4), dtype=torch.long)

@pytest.fixture
def tmp_data_dir(tmp_path):
    # Example temp dir for any file IO tests
    d = tmp_path / "data"
    d.mkdir()
    return d
