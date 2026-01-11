import json
import os
import string
from typing import Optional, Tuple
import torch
from tinygpt.configs.config import GPT2DataConfig


def save_training_checkpoint(
        cfg: GPT2DataConfig,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional,
        epoch: int,
        loss: float,
        global_step: int,
        ddp_initialized: bool,
):
    """Save full checkpoint (model + optimizer + data progress)"""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        cfg.checkpoint_dir,
        f"ckpt_gpt2_epoch_{epoch}_{global_step}.pt"
    )

    # Save DDP model (use model.module for DDP-wrapped models)
    # Save model weights
    # Step 1: Unwrap the COMPILED model → get uncompiled model (PyTorch 2.1+ ONLY uses _orig_mod)
    unwrapped_model = model._orig_mod  # critical: extract uncompiled model from OptimizedModule
    # Step 2: Unwrap DDP (if enabled) → get PURE vanilla GPT2 model (no wrappers left!)
    vanilla_model = unwrapped_model.module if ddp_initialized else unwrapped_model
    model_state = vanilla_model.state_dict()

    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(), # (**MOST IMPORTANT FOR RESUME - EASILY FORGOTTEN**)
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "loss": loss,
        "global_step": global_step
    }

    # Only save from rank 0 (avoid duplicate checkpoints)
    if cfg.rank == 0:
        torch.save(checkpoint, checkpoint_path)
