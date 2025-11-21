from data_loaders.gpt2_data_loader import make_dataloader, load_tokens
from config import GPT2DataConfig
import tiktoken

def test_gpt2_data_loader_default_input_corpus():
    # The default is the Shakespeare
    input_text_corpus, tokens = load_tokens(GPT2DataConfig)
    assert len(input_text_corpus) == 1115394 # total number of characters is 1,115,394
    assert input_text_corpus[0:15] == "First Citizen:\n"
    # "First Citizen" is a character name in several of Shakespeare's plays,
    # most prominently in Coriolanus, where he represents the voice of the Roman plebeians
    # who are discontented with the patrician rulers

    # GPT2 tokenizer is a BPE sub-word tokenizer
    assert len(tokens) == 338025 # 338,025 which is roughly 1/3 of 1,115,394, meaning the gpt2 tokenizer's compression rate is 1/3
    encoder = tiktoken.get_encoding("gpt2")
    assert tokens[0:4] == encoder.encode("First Citizen:\n")
    print(type(tokens)) # <class 'list'>

def test_gpt2_data_loader_default_x_y():
    dl = make_dataloader(GPT2DataConfig)
    x = dl["idx"]
    y = dl["target"]
    print(x.shape)
    print(y.shape)


