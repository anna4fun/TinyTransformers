import os
import pytest
from tinygpt.configs.config import GPT2DataConfig
from tinygpt.logger.setup_logger import setup_logger
from tinygpt.distributed_utils.setup_ddp_init import init_ddp


def test_ddp_setup():
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    device, ddp_rank, ddp_local_rank, ddp_world_size, master_process = init_ddp(ddp)
    cfg = GPT2DataConfig()
    logger = setup_logger(cfg = cfg, train_name = "gpt2-shakespeare-v2-DDP", local_rank = ddp_local_rank)
    # In test_ddp_setup.py
    logger.info(f"DDP device {device}")
    logger.info(f"DDP rank {ddp_rank}")
    logger.info(f"DDP local rank {ddp_local_rank}")
    logger.info(f"DDP world size {ddp_world_size}")
    logger.info(f"DDP master process {master_process}")  # Fixed typo (ddp_master_process → master_process)


if __name__ == "__main__":
    # Run all tests in the current script
    # -v: verbose output (optional, for clarity)
    # -x: stop on first failure (optional)
    exit_code = pytest.main(["-v", __file__])

    # Exit with pytest's exit code (0 = all pass, 1 = some fail, 2 = internal error)
    import sys
    sys.exit(exit_code)