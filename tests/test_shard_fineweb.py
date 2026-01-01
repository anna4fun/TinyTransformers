from tinygpt.configs.config import GPT2DataConfig
import torch
from pathlib import Path
import tiktoken
import numpy as np

def test_049file_legit():
    cfg = GPT2DataConfig()

    device = "mps"
    file_path = cfg.mps_fineweb_path
    assert file_path.exists(), f"ERROR: File does NOT exist → {file_path}"

    npy049_ds = np.load(file_path)
    print(npy049_ds.shape)
    print(npy049_ds.dtype)


# --------------------------
# Make the script executable
# --------------------------
if __name__ == "__main__":
    # Run the test when the script is called directly
    success = test_049file_legit()
    # Exit with non-zero code if test fails (for CI/scripting)
    import sys
    sys.exit(0 if success else 1)



