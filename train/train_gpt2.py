import code
import dataclasses
import time

import torch
from torch.multiprocessing import freeze_support
import sys
import swanlab
import os

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from tinygpt.checkpoints_utils.save_load_checkpoints import save_training_checkpoint
from tinygpt.distributed_utils.setup_ddp_init import init_ddp
from tinygpt.logger.setup_logger import setup_logger
from tinygpt.models.gpt2 import GPT2
from tinygpt.data_loaders.gpt2_data_loader import make_dataloader, make_ddp_dataloader
from tinygpt.configs.config import ExperimentConfig, GPT2DataConfig

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

@torch.no_grad()
def evaluate(model: GPT2, valid_loader: DataLoader, device, max_val_batches=100):
    # Total of 1000,000 seq in val_dl, pick 100 for validation every time, validate for 40 times
    model.eval()
    total_loss, total_samples = 0.0, 0
    valid_iter = iter(valid_loader)  # use a persistent iterator
    with torch.autocast(device_type=device, dtype=torch.float16):
        # No loss of loss metric accuracy (avg NLL is stable in FP16) + ~2x faster inference vs FP32.
        for i in range(max_val_batches):
            try:
                x, y = next(valid_iter)
            except StopIteration:
                break  # exit loop early if no more batches
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # extra speed
            _, loss = model(x,y) # this loss is the average negative log-likelihood(per sample in the batch)
            batch_size = x.size(0)
            total_loss += batch_size * loss.item()
            total_samples += batch_size
        val_loss = total_loss / max(1, total_samples)
        model.train()
    return val_loss # in case of 0 sample

def train_gpt2():
    # ----------------------
    # 1. Setup Device and DDP
    # ----------------------
    # DDP Initialization (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp_initialized = False
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    device, ddp_rank, ddp_local_rank, ddp_world_size, master_process = init_ddp(ddp)
    ddp_initialized = ddp and dist.is_initialized()
    if device.startswith("cuda"):
        device_type = "cuda"
    # todo: remember to update epoch
    epoch = 1

    # 2. Setup Model parameters and initialized tracking with SwanLab
    # ----------------------
    config = GPT2DataConfig(vocab_size=50304, batch_size=16, learning_rate=6e-4, device=device,
                            num_workers=16*2, ddp=ddp_initialized, resume_checkpoint=True)
    config_dict = dataclasses.asdict(config)
    logger = setup_logger(cfg = config, train_name = "gpt2-FineWeb-val100samples", local_rank = ddp_local_rank)

    # Complete deterministic
    # All ranks (GPUs) must use the same seed to avoid non-determinism in DDP
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # Disable non-deterministic CUDA ops (critical for GPU)
    # --------------------------
    torch.backends.cudnn.benchmark = False  # Disable speed optimizations for determinism
    torch.backends.cudnn.deterministic = True

    # --------------------------
    # Step 3: Seed DataLoader and Swanlab
    # --------------------------
    # Create a seeded generator for the DataLoader
    dl_generator = torch.Generator()
    dl_generator.manual_seed(config.seed)
    # TODO: modify dataloader for DDP: process_rank=ddp_rank, num_processes=ddp_world_size
    dl = make_ddp_dataloader(config, g=dl_generator)
    train_dl, valid_dl, test_dl = dl["train_dl"], dl["val_dl"], dl["test_dl"]

    # Initialize SwanLab only on master node
    swanlab_run = None
    if master_process:
        swanlab_run = swanlab.init(
            project="gpt2-training",  # Your project name
            experiment_name="gpt2-FineWeb-prod-2-4k",
            config=config_dict,  # Log hyperparameters
            mode="local",  # Use local mode (no cloud sync)
            description = "GPT-2 124M experiment training on FineWeb-edu dataset",
            tags = ["GPT2", "Experiment", "FineWeb", "small dataset"],
        )
        logger.info(f"Master Process {ddp_rank} initialized SwanLab successfully")
    # TODO: test DDP master process verification process
    logger.info(f"I am GPU {ddp_rank}")

    # ----------------------
    # 4. Model Setup (GPT2 + Optimizer + LR scheduler +  DDP + Compile)
    # ----------------------
    # Applies TensorFloat32 operation only to CUDA matrix multiplication operations
    # TF32 is the precision-reduced variant of FP32, but not half-precision(FP16/BF16)
    # It is a specialized 19-bit floating-point format designed by NVIDIA for accelerating AI workloads, distinct from both 32-bit single-precision (FP32) and 16-bit half-precision (FP16/BF16) formats
    torch.set_float32_matmul_precision('high')
    # Create the model and move model to the GPU device
    model = GPT2(config)
    model.to(device)
    # Optimizer sits outside the iteration loop
    # start with the max lr=6e-4
    optimizer = model.configure_optimizers(weight_decay=0.1, lr=model.config.learning_rate, device=device)

    # learning rate scheduler
    # Number of steps to go through 1 epoch
    max_steps = 19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
    warmup_steps = 715
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=model.config.learning_rate * 0.1)

    # Initialize training progress variables (defaults for scratch training)
    start_step = 0
    max_steps = 501  # TODO: change steps with max_step in PROD

    # Resume from checkpoints or start fresh
    checkpoint_name = "ckpt_gpt2_epoch_1_2000.pt" # todo: change to desired ckpt
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_name)
    if config.resume_checkpoint and os.path.exists(checkpoint_path):
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Restore ALL critical states
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training progress
        start_epoch = checkpoint["epoch"]
        start_step = checkpoint["global_step"] + 1 # iterate to the next dataloder batch
        last_loss = checkpoint["loss"]
        logger.info(f"Resumed training from epoch {start_epoch}, global step {start_step}, last loss: {last_loss:.4f}")
    elif config.resume_checkpoint and not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        sys.exit(0)
    else:
        print("Starting training from scratch")

    # Compile model speed up training, compilation itself consume some time
    # Efficiency gain comes from reducing Python overhead and GPU R/W
    ## note: compile is default, unless you are debugging and want to save compile time
    # Due to incompatibility of torch.compile() for MPS devices, only allow it for GPU devices.
    # Critical DDP Fix: Wrap model with DDP (MISSING IN ORIGINAL CODE → CRASH!)
    if ddp_initialized:
        model = DDP(model, device_ids=[ddp_local_rank])
        logger.info(f"DDP model wrapped successfully for GPU {ddp_rank}")

    # Compile model (safe for CUDA only)
    if device == "cuda":
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile()")

    # ----------------------
    # 5. Gradient accumulation
    # ----------------------
    # Gradient Accumulations to fit GPT3 paper - GPT3 125M parameter model batch size 0.5M
    total_batch_size = 524288  # 2**19, closest with 0.5M
    assert total_batch_size % config.batch_size * config.block_size == 0
    gradient_accumulation_steps = int(total_batch_size / (config.batch_size * config.block_size)) # 32
    logger.info(f"✅ Total batch size: {total_batch_size} | Gradient Accum Steps: {gradient_accumulation_steps}")
    if master_process:
        logger.info(f"Total desired batch size: {total_batch_size}")
        logger.info(f"=> Calculated gradient accumulation steps: {gradient_accumulation_steps}")

    # ----------------------
    # 6. Training Tracking Variables and load from resume state
    # ----------------------
    best_valid_loss = float('inf')
    start_time = time.time()  # Track total training time
    log_every = model.config.eval_interval
    train_iter = iter(train_dl) # use a persistent iterator
    model.train() # todo: what does .train() do?
    logger.info(f"Starting training from {start_step} steps to {max_steps} steps")

    # ----------------------
    # 7. Main Training Loop
    # ----------------------
    for i in range(start_step, max_steps):
        t0 = time.time() # current time in seconds since the Unix epoch
        # Forward and Backward Path
        # !! start with zero gradients
        optimizer.zero_grad()
        # Gradient accumulation to accumulate 0.5M token's of gradients every update step, remember to normalize the loss by grad_accum_steps due to "sum of average != overall average"
        loss_accum = 0.0

        for micro_step in range(gradient_accumulation_steps):
            # Load batch
            # x,y = next(iter(train_dl)) # Problem: You're using next(iter(train_dl)) inside the training loop, which re-initializes the DataLoader iterator every iteration. This is extremely inefficient and likely causing a CPU bottleneck that masks GPU speedups.
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)
            x = x.to(device)
            y = y.to(device)

            # Forward
            # Autocast to BF16 in the forward path (BF16 only available on Ampere architecture GPUs)
            with torch.autocast(device_type = device,dtype=torch.bfloat16):
                logits, loss = model(x, y)
                # Pause training to inspect state, e.g. the precision of logits should show float16, while wte.weights are float32
                # code.interact(local=locals())
            # Exit autocast before the backward path

            # Backward
            # TODO: if ddp, only synchronize the gradients when it is the last step of the gradient accumulated batch
            # Normalize gradient_accumulation_steps before loss.backward()
            loss = loss / gradient_accumulation_steps
            # Get the raw gradients and deposit(sum) them into params.grad
            loss.backward()
            # track leave nodes
            loss_accum += loss.detach().item()

        # Clip the global norm of the gradients at 1.0 (GPT3)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update the parameters: this is essentially w -= lr * w.grad()
        optimizer.step()

        # The synchronize function blocks the CPU thread until all pending MPS operations finish executing on the GPU
        if device =="cuda":
            torch.cuda.synchronize(device)
        elif device == "mps":
            torch.mps.synchronize()
            # CPU requires no synchronization (operations are synchronous by default)

        # ----------------------
        # Log Metrics (SwanLab)
        # ----------------------
        dt = time.time()- t0 # time difference in seconds
        # tokens per seconds
        B, T = x.shape
        token_per_sec = (B*T*gradient_accumulation_steps)/dt
        # log running time and train loss every iteration
        swanlab.log({
            "Loss/train_loss": loss_accum, # loss.item(), with gradient accumulation, each individual loss would be overall batch's loss/gradient_accumulation, so we should not use this,
            "Time/current_step_train_time": dt,
            "Time/token_per_sec": token_per_sec,
            "Norm": round(norm.item(), 6),
            "Learning Rate": float(scheduler.get_last_lr()[0]),
        }, step=i)

        if (i % log_every == 0 or i == 0 or i == max_steps-1) and master_process:
            # SWITCH TO SPEED MODE FOR EVAL: enable benchmark + disable strict determinism
            t3 = time.time()
            torch.backends.cudnn.benchmark = True
            eval_loss = evaluate(model, valid_dl, device, max_val_batches=100)

            # Update the best validation loss
            if eval_loss < best_valid_loss:
                best_valid_loss = eval_loss
            eval_time = time.time() - t3
            # Log train and validation loss
            swanlab.log({
                "Loss/valid_loss": eval_loss,
                "Loss/best_valid_loss": best_valid_loss,
                "Time/eval_time": eval_time,
            }, step=i)
            # SWITCH BACK TO DETERMINISTIC MODE FOR TRAINING (critical!)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        # Learning Rate scheduler update (happens after optimizer step)
        scheduler.step()

        # ----------------------
        # 6. Save model checkpoints per 500 steps
        # To pause/resume training:
        # * Save data progress (e.g., last processed shard + batch index) alongside model checkpoints.
        # * Save model weights, optimizer state, scheduler state, and data state (shard/batch).
        # ----------------------
        if (i % log_every == 0 or i == 0 or i == max_steps-1) and master_process:
            save_training_checkpoint(cfg=config, model=model, optimizer=optimizer, scheduler=scheduler,
                                     epoch=epoch, loss=loss_accum, global_step=i, ddp_initialized=ddp_initialized)

        # Prevent VRAM fragmentation: clean cache every 100 steps
        if i % 100 == 0:
            torch.cuda.empty_cache()

    # ----------------------
    # 7. Final Logging and clean up
    # ----------------------
    total_training_time = time.time() - start_time
    swanlab.log({"total_training_time": total_training_time})
    swanlab.finish()  # Mark run as finished in SwanLab
    logger.info(f"Training finished! Total time: {total_training_time:.2f}s")
    # Important: clean up Dataloader multi-processor
    if 'train_dl' in locals():
        del train_dl
    if 'valid_dl' in locals():
        del valid_dl
    if 'test_dl' in locals():
        del test_dl
    torch.cuda.empty_cache()  # clean cache
    if ddp_initialized:
        dist.destroy_process_group()
        logger.info("DDP process group destroyed successfully")


if __name__ == '__main__':
    freeze_support()
    if sys.platform == 'darwin':
        torch.multiprocessing.set_start_method('spawn', force=True)
    # The try/finally block will ensure the shutdown commands run
    # when training success/ crash with error/ manually stopped/ hit max step or epoch
    try:
        train_gpt2()
    finally:
        # time.sleep(5)  # wait for disk write
        # only the master process execute the shutdown command
        # if int(os.environ.get('RANK', 0)) == 0:
        print("Initiating cloud GPU instance shutdown...")
        # sys.exit(0)
