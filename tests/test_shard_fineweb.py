import pytest

from tinygpt.configs.config import GPT2DataConfig
import torch
from pathlib import Path
import tiktoken
import numpy as np

def test_048file_legit():
    cfg = GPT2DataConfig()

    device = "cpu"
    file_path = cfg.gpu_audodl_fineweb_path / "train/fineweb_edu_0048.npy"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        file_path = cfg.mps_fineweb_path

    # file exist
    assert file_path.exists(), f"ERROR: File does NOT exist → {file_path}"

    npy048_ds = np.load(file_path)

    # size match
    assert npy048_ds.shape[0] == 200000000, f"ERROR: npy048_ds shape is {npy048_ds.shape}"

    # convert ds into tensor and verify vocab_size
    npy048_ts = torch.tensor(npy048_ds, dtype=torch.long)
    vocab_size = len(torch.unique(npy048_ts))
    assert vocab_size == 49853, f"ERROR: vocab_size is {vocab_size}"
    # 99.2% of the total GPT2 vocab size 50257

    # extract the first 64 for decode
    sample_tokens = npy048_ts[:64]
    assert torch.equal(sample_tokens, torch.tensor([  447,   247,  2168,   290,    11, 14572,    11,  2968,   355,   428,
         4866,   318,   422,   262,   665, 44659,  7319, 13570,    13,  3771,
          926,   813, 18542,    11,   351,   281, 18542,  3002,    11, 27561,
        28537,    11,   351,   607, 20450,    11, 44681,   257, 10657, 10686,
           13,   383,  1479,  2166,   886, 20189,  4909,   257,   281, 16882,
        40189,   564,   246,  2514, 12091, 10452,    13,   554, 44827,   286,
         1611,  3241,  1141,  8526]))

    # tiktoken decoder
    decoder = tiktoken.get_encoding("gpt2")
    test_tokens_ls = sample_tokens.tolist()
    decoded_text = decoder.decode(test_tokens_ls)
    assert decoded_text == "’ series and, presumably, popular as this copy is from the twelfth thousand printing. Prettily illustrated, with an illustrated cover, depicting Florence, with her lamp, tending a wounded soldier. The free front endpaper contains a an ink inscription ‘To Jane Small. In remembrance of kind attention during illness"


if __name__ == "__main__":
    # Run all tests in the current script
    # -v: verbose output (optional, for clarity)
    # -x: stop on first failure (optional)
    exit_code = pytest.main(["-v", __file__])

    # Exit with pytest's exit code (0 = all pass, 1 = some fail, 2 = internal error)
    import sys
    sys.exit(exit_code)



