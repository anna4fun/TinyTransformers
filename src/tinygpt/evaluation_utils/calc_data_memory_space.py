import os
import numpy as np

def human_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def estimate(npy_path, batch_size, block_size, stride=None, include_next_token=True):
    arr = np.load(npy_path, mmap_mode="r")  # avoids loading whole file into RAM
    n_tokens = int(arr.size)
    itemsize = int(arr.dtype.itemsize)
    file_size = os.path.getsize(npy_path)

    # seq_len tokens consumed per training example (x needs block_size tokens,
    # and y is usually the same tokens shifted by 1, which requires one extra token available).
    seq_len = block_size + (1 if include_next_token else 0)

    tokens_per_batch = batch_size * block_size
    bytes_per_batch_if_stored = tokens_per_batch * itemsize

    if stride is None:
        # Non-overlapping chunks of length block_size (rough approximation)
        est_sequences = n_tokens // block_size
    else:
        if n_tokens < seq_len:
            est_sequences = 0
        else:
            est_sequences = 1 + (n_tokens - seq_len) // stride

    est_batches = est_sequences // batch_size if batch_size > 0 else 0

    print(f"File: {npy_path}")
    print(f"File size on disk: {human_bytes(file_size)}")
    print(f"Array dtype: {arr.dtype} ({itemsize} bytes/token)")
    print(f"Total tokens in array: {n_tokens:,}")
    print(f"Implied raw token storage: {human_bytes(n_tokens * itemsize)} (excludes .npy header)")
    print()
    print(f"batch_size={batch_size}, block_size={block_size}")
    print(f"Tokens per batch (x): {tokens_per_batch:,}")
    print(f"Bytes per batch (if you stored x contiguously): {human_bytes(bytes_per_batch_if_stored)}")
    print()
    if stride is None:
        print("Sampling assumption: non-overlapping blocks (rough)")
    else:
        print(f"Sampling assumption: sliding windows with stride={stride}, seq_len={seq_len}")
    print(f"Estimated sequences: {est_sequences:,}")
    print(f"Estimated full batches: {est_batches:,}")

if __name__ == "__main__":
    # Example usage:
    # estimate("fineweb_edu_0049.npy", batch_size=16, block_size=1024, stride=1024)
    # estimate("fineweb_edu_0049.npy", batch_size=16, block_size=1024, stride=1)  # maximal overlap
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("npy_path")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--block-size", type=int, required=True)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--no-next-token", action="store_true")
    args = p.parse_args()

    estimate(
        args.npy_path,
        batch_size=args.batch_size,
        block_size=args.block_size,
        stride=args.stride,
        include_next_token=not args.no_next_token,
    )
