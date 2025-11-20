# tests/conftest.py
from __future__ import annotations
import os
import random
import torch
import pytest
from dataclasses import dataclass

from config import DataConfig, ModelConfig
from data_loaders.data_loader import make_dataloaders
from gpt import GPTLanguageModel


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


@dataclass
class SelfAttentionHeadTestSetup:
    data_config: DataConfig
    model_config: ModelConfig
    model: GPTLanguageModel
    x: torch.Tensor
    hs: int
# scope="module" makes the fixture run once per test file, not before every test → faster.
# The returned dict can be reused the same setup values across tests.
@pytest.fixture(scope="module")
def self_attention_head_setup() -> SelfAttentionHeadTestSetup:
    """Shared setup for testing SelfAttention subfunctions."""
    data_config = DataConfig(block_size=15, batch_size=8, val_frac=0.01, seed=42, shuffle=True)
    bundle = make_dataloaders(data_config)
    vocab_size = bundle["vocab_size"]
    model_config = ModelConfig(vocab_size=vocab_size, n_embd=4)
    # first load the GPT main model, we need it for the embedding
    model = GPTLanguageModel(model_config, data_config)
    xb, yb = next(iter(bundle["train_loader"]))
    te, pe = model.token_position_embedding(xb)
    x = te + pe  # x is token + position embedded xb
    return SelfAttentionHeadTestSetup(
        data_config=data_config,
        model_config=model_config,
        model=model,
        x=x,
        hs=10
    )

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
