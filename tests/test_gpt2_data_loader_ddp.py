import os
import pytest
import torch
from tinygpt.data_loaders.gpt2_data_loader import ResumableShardedLMSequenceDataset, logger, make_ddp_dataloader
from tinygpt.configs.config import GPT2DataConfig
from tinygpt.distributed_utils import init_ddp
from tinygpt.logger.setup_logger import setup_logger
import torch.distributed as dist


def test_resumeable_dataset():
    ddp_initialized = False
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    device, ddp_rank, ddp_local_rank, ddp_world_size, master_process = init_ddp(ddp)
    ddp_initialized = ddp and dist.is_initialized()

    try:
        if device == 'mps':
            pytest.skip("Skipping test on MPS device")
            return
        else:
            config = GPT2DataConfig(vocab_size=50304, batch_size=16, learning_rate=6e-4, device=device,
                                    num_workers=2) # reduce num_workers to avoid OOM
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

            # Dataset Initialization
            logger.info("now do the dataset class test")
            ds = ResumableShardedLMSequenceDataset(cfg=config, split="test")
            shard_paths_test = ds.shard_paths
            assert len(shard_paths_test) == 1, "Test dataset should have 1 shard"
            logger.info(f"Test dir Shard path {shard_paths_test}")
            ds = ResumableShardedLMSequenceDataset(cfg=config, split="train")
            shard_paths_train = ds.shard_paths
            assert len(shard_paths_train) == 49, "Train dataset should have 49 shards"
            # successfully tests
            # logger.info(f"Train dir Shard path {shard_paths_train}")
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            dl_generator = torch.Generator()
            dl_generator.manual_seed(config.seed)
            # create dataloader objects
            dl = make_ddp_dataloader(cfg=config, g=dl_generator)
            train_dl = dl["train_dl"]
            valid_dl = dl["val_dl"]
            test_dl = dl["test_dl"]
            # Train data loder
            # Safe move
            try:
                train_iter = iter(train_dl)
                x, y = next(train_iter)
            except StopIteration:
                pytest.fail("Train DataLoader is empty!")

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Confirm x and y are of shape (B, T)
            # the following code is not efficient
            # assert torch.equal(torch.tensor(x.shape), torch.tensor([config.batch_size, config.block_size]))
            # assert torch.equal(torch.tensor(y.shape), torch.tensor([config.batch_size, config.block_size]))
            assert x.shape == (config.batch_size, config.block_size), \
                f"X shape mismatch: expected {(config.batch_size, config.block_size)}, got {x.shape}"
            assert y.shape == (config.batch_size, config.block_size), \
                f"Y shape mismatch: expected {(config.batch_size, config.block_size)}, got {y.shape}"

            # Confirm 1st and 2nd sequence are not the same
            logger.info(f"1st row of X: {x[0, 0:5]}")
            logger.info(f"2nd row of X: {x[1, 0:5]}")
            logger.info(f"16th  row of X: {x[15, 0:5]}")

            # Confirm x and y are 1 token off
            assert torch.equal(x[0, 1:10], y[0, 0:9]), "X and Y are not 1 token offset" # Failed:because I compare the wrong part of x and y
            # # first print
            logger.info(f"Debug - X[0,1:10]: {x[0, 1:10].cpu().numpy()}")
            logger.info(f"Debug - Y[0,0:9]: {y[0, 0:9].cpu().numpy()}")
            # Confirm x row 1 and x row 2 are continuous within a batch
            assert torch.equal(x[1,0], y[0, config.block_size-1]), "x row 1 and x row 2 are continuous within a batch"
            logger.info(f"x row 1 and x row 2 are continuous within a batch")

            # Valid data loader
            try:
                val_iter = iter(valid_dl)
                vx, vy = next(val_iter)
            except StopIteration:
                pytest.fail("Valid DataLoader is empty!")

            assert vx.shape == (config.batch_size, config.block_size), \
                f"X shape mismatch: expected {(config.batch_size, config.block_size)}, got {vx.shape}"
            assert vy.shape == (config.batch_size, config.block_size), \
                f"Y shape mismatch: expected {(config.batch_size, config.block_size)}, got {vy.shape}"
            # Confirm 1st and 2nd sequence are not the same
            logger.info(f"1st row of X: {vx[0, 0:5]}")
            logger.info(f"2nd row of X: {vx[1, 0:5]}")
            # Confirm x and y are 1 token off
            assert torch.equal(vx[0, 1:10], vy[0, 0:9]), "X and Y are not 1 token offset"
            logger.info(f"Valid set passed")

            # TODO: Test save _save_resume_state
            # _save_resume_state()
            # TODO: Test load_resume_state

    finally:
        # Important: clean up Dataloader multi-processor
        if 'train_dl' in locals():
            del train_dl
        if 'test_dl' in locals():
            del test_dl
        torch.cuda.empty_cache() # clean cache

        if ddp_initialized:
            dist.destroy_process_group()
            logger.info("DDP process group destroyed successfully")


if __name__ == '__main__':
    # Run all tests in the current script
    # -v: verbose output (optional, for clarity)
    # -x: stop on first failure (optional)
    exit_code = pytest.main(["-v", "-x", "-s", "--disable-warnings", __file__])

    # Exit with pytest's exit code (0 = all pass, 1 = some fail, 2 = internal error)
    import sys
    sys.exit(exit_code)

