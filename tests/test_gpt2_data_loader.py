import pytest
import torch

from tinygpt.data_loaders.gpt2_data_loader import make_dataloader, load_tokens, train_valid_split
from tinygpt.configs.config import GPT2DataConfig

def test_gpt2_data_loader_default_input_corpus():
    # The default is the Shakespeare
    input_text_corpus, tokens, encoder = load_tokens(GPT2DataConfig)
    assert len(input_text_corpus) == 1115394 # total number of characters is 1,115,394
    assert input_text_corpus[0:15] == "First Citizen:\n"
    # "First Citizen" is a character name in several of Shakespeare's plays,
    # most prominently in Coriolanus, where he represents the voice of the Roman plebeians
    # who are discontented with the patrician rulers

    # GPT2 tokenizer is a BPE sub-word tokenizer
    assert len(tokens) == 338025 # 338,025 which is roughly 1/3 of 1,115,394, meaning the gpt2 tokenizer's compression rate is 1/3
    assert tokens[0:4] == encoder.encode("First Citizen:\n")
    print(type(tokens)) # <class 'list'>


def test_gpt2_data_loader_split():
    tokens = torch.arange(0, 20)
    # the default split_frac = 0.9
    train, valid = train_valid_split(tokens)
    assert len(train) == len(tokens)*0.9
    assert len(valid) == len(tokens)*0.1
    assert torch.equal(valid, torch.arange(0.9*20, 20))
    # try different split_frac = 0.4
    train, valid = train_valid_split(tokens, split_frac=0.4)
    assert len(train) == len(tokens)*0.4
    assert len(valid) == len(tokens)*0.6
    assert torch.equal(train, torch.arange(0, 0.4*20))


def test_gpt2_data_loader_default_x_y():
    dl = make_dataloader(GPT2DataConfig)
    # Train
    train_dl = dl["train_dl"]
    print(type(train_dl))
    x, y = next(iter(train_dl))
    # Confirm x and y are of shape (B, T)
    assert torch.equal(torch.tensor(x.shape), torch.tensor([GPT2DataConfig.batch_size, GPT2DataConfig.block_size]))
    assert torch.equal(torch.tensor(y.shape), torch.tensor([GPT2DataConfig.batch_size, GPT2DataConfig.block_size]))
    first_x = next(iter(x))
    first_y = next(iter(y))
    assert torch.equal(torch.tensor(first_x.shape), torch.tensor([GPT2DataConfig.block_size]))
    print(first_x[:20])
    print(first_y[:20])
    # confirm that y is x shift left by one token
    assert torch.equal(first_x[1:19], first_y[0:18])

    # Valid
    valid_dl = dl["valid_dl"]
    print(type(valid_dl))
    xv, yv = next(iter(valid_dl))
    # Confirm x and y are of shape (B, T)
    assert torch.equal(torch.tensor(xv.shape), torch.tensor([GPT2DataConfig.batch_size, GPT2DataConfig.block_size]))
    assert torch.equal(torch.tensor(yv.shape), torch.tensor([GPT2DataConfig.batch_size, GPT2DataConfig.block_size]))
    first_xv = next(iter(xv))
    first_yv = next(iter(yv))
    assert torch.equal(torch.tensor(first_x.shape), torch.tensor([GPT2DataConfig.block_size]))
    print(first_xv[500:520])
    print(first_yv[500:520])
    # confirm that y is x shift left by one token
    assert torch.equal(first_x[501:520], first_y[500:519])


if __name__ == "__main__":
    # Run all tests in the current script
    # -v: verbose output (optional, for clarity)
    # -x: stop on first failure (optional)
    exit_code = pytest.main(["-v", __file__])

    # Exit with pytest's exit code (0 = all pass, 1 = some fail, 2 = internal error)
    import sys
    sys.exit(exit_code)
