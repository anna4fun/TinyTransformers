import json
import os
from typing import Optional, Tuple
import torch
from tinygpt.configs.config import GPT2DataConfig


def save_training_checkpoint(
        cfg: GPT2DataConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional,
        epoch: int,
        loss: float,
        global_step: int,
):
    """Save full checkpoint (model + optimizer + data progress)"""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        cfg.checkpoint_dir,
        f"ckpt_gpt2_epoch_{epoch}_{global_step}.pt"
    )

    # Save DDP model (use model.module for DDP-wrapped models)
    # Save model weights
    model_state = model.module.state_dict() if cfg.ddp else model.state_dict()

    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(), # (**MOST IMPORTANT FOR RESUME - EASILY FORGOTTEN**)
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "cfg": cfg,
        "loss": loss,
        "global_step": global_step
    }

    # Only save from rank 0 (avoid duplicate checkpoints)
    if cfg.rank == 0:
        torch.save(checkpoint, checkpoint_path)

def load_training_checkpoint(
        cfg: GPT2DataConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
) -> Tuple[int, int, int, float]:
    # todo: load specific checkpoints
    """Load checkpoint (returns resume_shard_idx, resume_seq_idx, epoch, loss)"""
    if not cfg.resume_checkpoint or not os.path.exists(cfg.resume_checkpoint):
        print("No checkpoint found — starting fresh")
        return 0, 0, 0, 0.0

    # todo: change cfg.resume_checkpoint to specific checkpoints
    checkpoint = torch.load(cfg.resume_checkpoint, map_location=f"cuda:{cfg.rank}")

    # Load model (DDP-aware)
    if cfg.ddp:
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer/scheduler
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Resume state
    resume_shard_idx = checkpoint["resume_shard_idx"]
    resume_seq_idx = checkpoint["resume_seq_idx"]
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Resumed from checkpoint: shard {resume_shard_idx}, epoch {epoch}, loss {loss:.4f}")
    return resume_shard_idx, resume_seq_idx, epoch, loss