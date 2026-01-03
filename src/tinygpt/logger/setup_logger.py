import os
import sys
import logging
from tinygpt.configs.config import GPT2DataConfig


# Add GPU rank to console logs (via filter)
class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True

# -------------------------- Logger Configuration --------------------------
def setup_logger(cfg: GPT2DataConfig, train_name: "fineweb_training") -> logging.Logger:
    """
    Configure a global logger for DDP training:
    - Rank 0: Writes to file + console
    - Other ranks: Only writes to console (avoids duplicate logs)
    - Logs include timestamps, GPU rank, and severity
    """
    # Create logger
    logger = logging.getLogger(train_name)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for more verbosity
    logger.propagate = False  # Avoid duplicate logging

    # Clear existing handlers (prevent duplicates on resume)
    if logger.handlers:
        logger.handlers.clear()

    # Define log format (timestamp | GPU rank | level | message)
    log_format = logging.Formatter(
        "%(asctime)s | GPU %(rank)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler (all ranks see this)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)

    console_handler.addFilter(RankFilter(cfg.rank))
    logger.addHandler(console_handler)

    # 2. File Handler (only rank 0 writes to file to avoid duplicates)
    train_file = "training_{}.log".format(train_name)
    if cfg.rank == 0:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        log_file_path = os.path.join(cfg.checkpoint_dir, train_file)
        # Append mode (preserve logs across resume)
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(log_format)
        file_handler.addFilter(RankFilter(cfg.rank))
        logger.addHandler(file_handler)

    # Suppress unwanted logs from other libraries (e.g., datasets/huggingface)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("torch.distributed").setLevel(logging.WARNING)

    return logger
