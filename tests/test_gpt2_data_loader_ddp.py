import os
import pytest
import torch

from tinygpt.data_loaders.gpt2_data_loader import ResumableShardedLMSequenceDataset, logger, make_ddp_dataloader
from tinygpt.configs.config import GPT2DataConfig
from tinygpt.distributed_utils import init_ddp
from tinygpt.logger.setup_logger import setup_logger


def test_resumeable_dataset():
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    device, ddp_rank, ddp_local_rank, ddp_world_size, master_process = init_ddp(ddp)
    if device == 'mps':
        pass
    else:
        config = GPT2DataConfig(vocab_size=50304, batch_size=16, learning_rate=6e-4, device=device,
                                num_workers=16*2)
        logger = setup_logger(cfg=config, train_name="test_ddp_dataloader", local_rank=ddp_local_rank)
        logger.info("Start a new round")
        logger.info("Manual test file read")
        split_dir = config.gpu_audodl_fineweb_path / "train"
        assert split_dir.exists(), f"ERROR: File does NOT exist → {split_dir}"
        # List and sort shards numerically (shard_000.npy < shard_001.npy)
        shard_files = list(split_dir.glob("*.npy"))
        assert len(shard_files) == 49
        logger.info(shard_files[30])
        split_dir = config.gpu_audodl_fineweb_path / "test"
        assert split_dir.exists()
        shard_files = list(split_dir.glob("*.npy"))
        assert len(shard_files) == 1
        logger.info(shard_files[0])
        logger.info("now do the dataset class test")
        # The following code gets killed when running on CPU (2G), possibly due to OOM

        ds = ResumableShardedLMSequenceDataset(cfg=config, split="test")
        shard_paths_test = ds.shard_paths
        assert len(shard_paths_test) == 1
        logger.info(f"Test dir Shard path {shard_paths_test}")
        ds = ResumableShardedLMSequenceDataset(cfg=config, split="train")
        shard_paths_train = ds.shard_paths
        assert len(shard_paths_train) == 49
        # successfully tests
        # logger.info(f"Train dir Shard path {shard_paths_train}")

        torch.cuda.manual_seed_all(config.seed)
        dl = make_ddp_dataloader(cfg=config)
        train_dl = dl["train_dl"]
        test_dl = dl["test_dl"]
        x, y = next(iter(train_dl))
        x = x.to(device) # x, y should be (16, 1024) tensors
        y = y.to(device)
        logger.info(f"X shape: {x.shape}")
        logger.info(f"y shape: {y.shape}")
        # Confirm x and y are of shape (B, T)
        assert torch.equal(torch.tensor(x.shape), torch.tensor([config.batch_size, config.block_size]))
        assert torch.equal(torch.tensor(y.shape), torch.tensor([config.batch_size, config.block_size]))
        # logger.info(f"Shard token counts {ds.shard_token_counts}")
        # logger.info(f"Total tokens {ds.total_tokens}")
        # logger.info(f"Shard start indices: {ds.shard_start_indices}")
        # logger.info(f"Total sequences {ds.total_sequences}")


if __name__ == '__main__':
    # Run all tests in the current script
    # -v: verbose output (optional, for clarity)
    # -x: stop on first failure (optional)
    exit_code = pytest.main(["-v", __file__])

    # Exit with pytest's exit code (0 = all pass, 1 = some fail, 2 = internal error)
    import sys

    sys.exit(exit_code)

