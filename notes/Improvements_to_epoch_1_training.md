## Data Loaders
1. The deterministic mode

    During the training epoch 1, I had placed the determinstic mode because I thought this would enable the DataLoader to load the same sequence everytime I resume from checkpoints.

    ```
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ```

    Turns out this is a mis-understanding:

    1. `torch.backends.cudnn.deterministic` forces cuDNN to use only deterministic algorithms for convolutions and related CUDA operations (conv/matmul), not Pytorch RNG or DataLoader shuffling. Flag it `True` only when you want reproducibility (i.e debuggin) with the downside of being slow.
    2. The seeded generator already control the shuffing of DataLoader.

    Disable the deterministic mode for speed without affecting data reproducibility. It's 5% faster (token per sec: pre 1.19e+5 to post 1.26e+5)  

    Note: the train/val split currently don't use the generator. add it.
    
    Note: `torch.backends.cudnn.benchmark` is designed to run a short benchmark to pick the fastest convolution algorithm for the given input shape. When `True`, benchmarks once per input shape then reuse the same fastest algorithm. Since GPT-2 doesn't have a convolution layer so this flag is not used, what GPT-2 and other transformers do use for matrix multiplication is cuBLAS and cuBLASLt.

    Conclusion: delete them all.


2. How to treat not enough tokens in the current shard: concatenation vs padding?

    Concatenation is correct, 
    Padding is not used in language model training because the model needs to learn patterns around padding and waste computation, padding tokens also affect loss. 

    Concatenation implementation options w.r.t speed and memory impact:
    1. (memory efficient) on-demand loading the next shard: with num_workers > 0, other works can continue while one loads
    2. (memory overhead) preloading, could be async or threaded. Only make sense when num_workers = 0, very slow I/O

    Understanding workers and impact on memory:
    **Each dataloder worker gets its own dataset instance.**

    The napkin math of size per shard:
    - 10B tokens ÷ 49 shards ≈ 204M tokens per shard
    - Each token: int64 = 8 bytes
    Which means the per shard size is 204M * 8 = 1624M bytes ~1.6GB in memory

    With my current setup:
    - num_workers=32 (from  config: 16*2)
    - 32 workers × 1.6GB = ~51GB (on-demand, loading current shard and only load the next shard when necessary,  1 shard per worker)
    - 32 workers × 3.2GB = ~102GB (preloading current and next sharded file, 2 shard per worker) 
    
    Preloading 2x memory usage and can cause OOM with multiple workers.
