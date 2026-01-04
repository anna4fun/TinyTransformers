import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, List
import torch
from datasets import tqdm, load_from_disk
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from tinygpt.configs.config import GPT2DataConfig
import tiktoken
import logging
import random
from tqdm import tqdm
from datasets import load_from_disk, Dataset as HFDataset  # Hugging Face Datasets
from tinygpt.logger.setup_logger import setup_logger

# ----- The GPT2 Data Loader ------
"""
Process:
1. Load the documents onto CPU
2. Tokenize documents with tiktoken, output is a long 1-D tensor
3. (Deprecated)Slice the long 1-D tensor into batches of X and Y without shuffling. Each X and Y are of size (B,T), each sequence of Y is one token left of each sequence of X.
3. Sampling B random starting points, and chunk B batches of tokens from the 1-D tensors.
4. Return a DataLoader containing x and y (both (B,T)) into the GPT language model 
Warning: do not move data to GPU yet, the entire corpus will be huge and it's better stay at CPU
"""

# -------------------------
# Logging
# -------------------------
# logger = logging.getLogger("lm_dataset")
# logger.setLevel(logging.DEBUG)          # Or INFO
# handler = logging.StreamHandler()
# formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)
#
# # prevent double logging if module imported multiple times
# logger.propagate = False
# Setup logger (critical: call after DDP rank is set)
# todo: move this to train_gpt2.py after DDP enabled dataloader is build and fully tested
cfg = GPT2DataConfig()
logger = setup_logger(cfg=cfg, train_name='shakespeare', local_rank=0)


# -------------------------
# Data Loader for single untokenized file
# -------------------------
@lru_cache(maxsize=1)
def load_corpus(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

@lru_cache(maxsize=1)
def load_tokens(cfg: GPT2DataConfig, text_path= None) -> Tuple[torch.Tensor, tiktoken.Encoding]:
    # 1. load the input text, the default input is the Tiny Shakespeare
    if text_path is None:
        corpus = load_corpus(cfg.shakes_text_path)
    else:
        corpus = load_corpus(text_path)

    # 2. tokenize the corpus with tiktoken
    encoder = tiktoken.get_encoding("gpt2")
    tokens = encoder.encode(corpus)  # a list
    return corpus, tokens, encoder

def train_valid_split(tokens: torch.Tensor, split_frac: float=0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(tokens)
    split = int(n * split_frac)
    train = tokens[:split]
    valid = tokens[split:]
    return train, valid

class LMSequenceDataset(Dataset):
    """
    Language-model dataset that returns a (x, y) tuple, where x, y are both (block_size, ) tensors
    from a 1D token tensor.
    Given starting index i, x is tokens[i:i+T], y is tokens[i+1:i+1+T].
    """

    # global counter for logging
    log_counter = 0
    max_log_calls = 1  # only log the first 1 calls to avoid spam

    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

        if len(self.tokens) < self.block_size + 1:
            raise ValueError("Token is not enough to build 1 block")
            # TODO: change to patch

    def __len__(self):
        # max starting index for generating full block sized X, Y pairs
        # max starting index range from (0, ..., len(token) - block_size - 1) inclusive
        return len(self.tokens) - self.block_size

    def __getitem__(self, start_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[start_idx:   start_idx+self.block_size]
        y = self.tokens[start_idx+1: start_idx+self.block_size+1]
        # -------- Logging Block ----------
        # Log only first few calls to avoid 10k lines of debug spam
        if LMSequenceDataset.log_counter < LMSequenceDataset.max_log_calls:
            logger.debug(
                f"[Dataset] __getitem__ call #{LMSequenceDataset.log_counter}\n"
                f"  start_idx      = {start_idx}\n"
                f"  x slice        = [{start_idx}:{start_idx + self.block_size}] "
                f"(shape: {tuple(x.shape)})\n"
                f"  y slice        = [{start_idx + 1}:{start_idx + self.block_size + 1}] "
                f"(shape: {tuple(y.shape)})\n"
                f"  x[:8] preview  = {x[:8].tolist()}\n"
                f"  y[:8] preview  = {y[:8].tolist()}"
            )
            LMSequenceDataset.log_counter += 1
        # ---------------------------------
        return x, y

# Wrap DataSet object containing pairs of 1-D x and y into a DataLoader
def make_dataloader(cfg: GPT2DataConfig,
                    text_path=None,
                    train_frac=0.9,
                    num_workers=1) -> Dict[str,DataLoader]:
    corpus, tokens, encoder = load_tokens(cfg, text_path)
    tokens = torch.LongTensor(tokens)
    train_tokens, valid_tokens = train_valid_split(tokens, train_frac)
    train_dataset = LMSequenceDataset(train_tokens, cfg.block_size)
    valid_dataset = LMSequenceDataset(valid_tokens, cfg.block_size)

    torch.manual_seed(cfg.seed)
    # Train DataLoader
    train_dl = DataLoader(train_dataset,
                          batch_size=cfg.batch_size,
                          num_workers=num_workers,
                          shuffle=cfg.shuffle,
                          pin_memory=True,
                          drop_last=False,)

    # Valid DataLoader
    valid_dl = DataLoader(valid_dataset,
                          batch_size=cfg.batch_size,
                          num_workers=num_workers,
                          shuffle=cfg.shuffle,
                          pin_memory=True,
                          drop_last=False,)
    return dict(train_dl=train_dl,valid_dl=valid_dl, encoder=encoder)


# -------------------------- Sharded LM Dataset (Resumable + DDP) --------------------------
class ResumableShardedLMSequenceDataset(Dataset):
    """
    - Loads shards from pre-split train/test folders (no manual splitting)
    - DDP-compatible (unique data per GPU)
    - Trackable progress for resumable training
    """

    def __init__(self, cfg: GPT2DataConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split  # "train" or "test" (uses pre-split folders)
        self.block_size = cfg.block_size

        # Step 1: List all shard paths (sorted) from pre-split folder
        self.shard_paths = self._get_sorted_shard_paths()
        # Step 2: Precompute token offsets + total tokens (per shard)
        self.shard_token_counts, self.total_tokens = self._compute_shard_token_counts()
        # Step 3: Precompute valid sequence start indices (per shard)
        self.shard_start_indices = self._get_shard_start_indices()
        # Step 4: Total valid (x,y) pairs (full coverage)
        self.total_sequences = sum(len(indices) for indices in self.shard_start_indices.values())

        # State for streaming/caching
        self.current_shard_idx = -1
        self.cached_tokens = torch.LongTensor([])
        self.cached_shard_idx = -1

        # Resumable state (track progress)
        self.resume_shard_idx = 0  # Start from shard 0 by default
        self.resume_seq_idx = 0  # Start from first sequence in resume_shard_idx
        if cfg.resume_checkpoint:
            self._load_resume_state()

    def _get_sorted_shard_paths(self) -> List[str]:
        """Load shards from pre-split train/test folder (e.g., ./fineweb_edu/train/)"""
        split_dir = os.path.join(self.cfg.shard_root_dir, self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Pre-split folder not found: {split_dir}")

        # List and sort shards numerically (shard_000.parquet < shard_001.parquet)
        shard_files = [
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith((".parquet", ".jsonl")) and "shard" in f
        ]
        shard_files.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
        return shard_files

    def _compute_shard_token_counts(self) -> Tuple[Dict[int, int], int]:
        """
        Precompute:
        - shard_token_counts: {shard_idx: number of tokens in shard}
        - total_tokens: total tokens in the split (train/test)
        """
        shard_token_counts = {}
        total_tokens = 0
        for shard_idx, shard_path in enumerate(tqdm(self.shard_paths, desc=f"Counting {self.split} tokens")):
            # Load shard (lazy) and count tokens
            if shard_path.endswith(".parquet"):
                from datasets import Dataset as HFDataset
                shard = HFDataset.from_parquet(shard_path)
            elif shard_path.endswith(".jsonl"):
                from datasets import Dataset as HFDataset
                shard = HFDataset.from_json(shard_path)
            else:
                raise ValueError(f"Unsupported shard format: {shard_path}")

            token_count = sum(len(example[self.cfg.token_column]) for example in shard)
            shard_token_counts[shard_idx] = token_count
            total_tokens += token_count
        return shard_token_counts, total_tokens

    def _get_shard_start_indices(self) -> Dict[int, List[int]]:
        """
        Precompute valid start indices PER SHARD (for resumable training):
        - shard_start_indices: {shard_idx: list of start indices (relative to shard)}
        """
        shard_start_indices = {}
        global_token_offset = 0  # Global token index across all shards

        for shard_idx in range(len(self.shard_paths)):
            shard_token_count = self.shard_token_counts[shard_idx]
            # Valid start indices for this shard (relative to shard)
            max_local_start = shard_token_count - self.block_size - 1
            if max_local_start < 0:
                shard_start_indices[shard_idx] = []
                global_token_offset += shard_token_count
                continue

            # Local start indices (0 to max_local_start)
            local_start_indices = list(range(max_local_start + 1))
            shard_start_indices[shard_idx] = local_start_indices
            global_token_offset += shard_token_count

        return shard_start_indices

    def _load_shard_tokens(self, shard_idx: int) -> torch.LongTensor:
        """Load a single shard into a 1D tensor (cached)"""
        shard_path = self.shard_paths[shard_idx]
        from datasets import Dataset as HFDataset

        if shard_path.endswith(".parquet"):
            shard = HFDataset.from_parquet(shard_path)
        elif shard_path.endswith(".jsonl"):
            shard = HFDataset.from_json(shard_path)
        else:
            raise ValueError(f"Unsupported shard format: {shard_path}")

        all_tokens = []
        for example in shard:
            all_tokens.extend(example[self.cfg.token_column])
        return torch.LongTensor(all_tokens)

    def _get_tokens_from_shard(self, shard_idx: int, local_start: int, local_end: int) -> torch.Tensor:
        """Get tokens from a single shard (local indices)"""
        if self.cached_shard_idx != shard_idx:
            self.cached_tokens = self._load_shard_tokens(shard_idx)
            self.cached_shard_idx = shard_idx
        return self.cached_tokens[local_start:local_end]

    def _get_globally_unique_indices(self) -> List[int]:
        """
        DDP: Split sequence indices across GPUs (each GPU gets a unique subset)
        Returns list of global sequence indices for this GPU
        """
        all_sequences = []
        # Flatten all sequence indices (global_seq_idx = [shard_idx][local_seq_idx])
        for shard_idx in range(len(self.shard_paths)):
            for local_seq_idx in self.shard_start_indices[shard_idx]:
                all_sequences.append((shard_idx, local_seq_idx))

        # DDP: Split sequences across world_size GPUs (rank = current GPU ID)
        if self.cfg.ddp:
            # Ensure deterministic split
            random.seed(self.cfg.seed)
            all_sequences = sorted(all_sequences)
            # Assign sequences to GPU rank (round-robin)
            all_sequences = all_sequences[self.cfg.rank::self.cfg.world_size]

        # Filter for resume state (skip processed sequences)
        resume_sequences = []
        for shard_idx, local_seq_idx in all_sequences:
            if shard_idx < self.resume_shard_idx:
                continue
            if shard_idx == self.resume_shard_idx and local_seq_idx < self.resume_seq_idx:
                continue
            resume_sequences.append((shard_idx, local_seq_idx))

        return resume_sequences

    def __len__(self) -> int:
        """Number of sequences for THIS GPU (DDP-aware)"""
        return len(self._get_globally_unique_indices())

    def __getitem__(self, global_seq_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get (x,y) pair (DDP + resume-aware)"""
        # Map global_seq_idx to (shard_idx, local_seq_idx)
        unique_indices = self._get_globally_unique_indices()
        shard_idx, local_seq_idx = unique_indices[global_seq_idx]

        # Local start/end for x/y (relative to shard)
        local_x_start = local_seq_idx
        local_x_end = local_seq_idx + self.block_size
        local_y_start = local_seq_idx + 1
        local_y_end = local_seq_idx + self.block_size + 1

        # Get x (input) and y (target)
        x = self._get_tokens_from_shard(shard_idx, local_x_start, local_x_end)
        y = self._get_tokens_from_shard(shard_idx, local_y_start, local_y_end)

        # Cross-shard handling (if sequence spans shards)
        if len(x) < self.block_size:
            # Fetch remaining tokens from next shard
            next_shard_idx = shard_idx + 1
            remaining = self.block_size - len(x)
            x = torch.cat([x, self._get_tokens_from_shard(next_shard_idx, 0, remaining)])
            y = torch.cat([y, self._get_tokens_from_shard(next_shard_idx, 1, remaining + 1)])

        assert len(x) == self.block_size and len(y) == self.block_size
        return x, y

    def _save_resume_state(self, shard_idx: int, seq_idx: int):
        """Save resume state (last processed shard + sequence index)"""
        resume_state = {
            "resume_shard_idx": shard_idx,
            "resume_seq_idx": seq_idx,
            "shard_paths": self.shard_paths,
            "split": self.split
        }
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        state_path = os.path.join(self.cfg.checkpoint_dir, "resume_state.json")
        with open(state_path, "w") as f:
            json.dump(resume_state, f)

    def _load_resume_state(self):
        """Load resume state from checkpoint"""
        state_path = os.path.join(self.cfg.checkpoint_dir, "resume_state.json")
        if not os.path.exists(state_path):
            print(f"No resume state found at {state_path} — starting fresh")
            return

        with open(state_path, "r") as f:
            resume_state = json.load(f)

        self.resume_shard_idx = resume_state["resume_shard_idx"]
        self.resume_seq_idx = resume_state["resume_seq_idx"]
        print(f"Resumed from shard {self.resume_shard_idx}, sequence {self.resume_seq_idx}")


# -------------------------- DDP DataLoader Builder --------------------------
def make_ddp_dataloader(cfg: GPT2DataConfig) -> Dict[str, DataLoader]:
    """Build DDP-compatible DataLoaders with resumable training"""
    # Initialize DDP (call this once at the start of training)
    if cfg.ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")  # NCCL for GPU DDP

    # Create train/test datasets (pre-split folders)
    train_dataset = ResumableShardedLMSequenceDataset(cfg, split="train")
    test_dataset = ResumableShardedLMSequenceDataset(cfg, split="test")

    # DDP: Use DistributedSampler (critical for batch splitting across GPUs)
    def build_dl(dataset: ResumableShardedLMSequenceDataset, split: str) -> DataLoader:
        sampler = DistributedSampler(
            dataset,
            num_replicas=cfg.world_size,
            rank=cfg.rank,
            shuffle=cfg.shuffle,
            seed=cfg.seed
        ) if cfg.ddp else None

        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,  # Overrides shuffle (use sampler for DDP)
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if cfg.num_workers > 0 else False
        )

    train_dl = build_dl(train_dataset, "train")
    test_dl = build_dl(test_dataset, "test")

    return {
        "train_dl": train_dl,
        "test_dl": test_dl,
        "train_dataset": train_dataset,  # For checkpointing progress
        "test_dataset": test_dataset
    }
