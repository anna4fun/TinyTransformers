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

from tinygpt.checkpoints_utils.save_load_checkpoints import save_training_checkpoint, load_training_checkpoint
from tinygpt.distributed_utils.setup_ddp_init import init_ddp
from tinygpt.logger.setup_logger import setup_logger
from tinygpt.models.gpt2 import GPT2
from tinygpt.data_loaders.gpt2_data_loader import make_dataloader, make_ddp_dataloader
from tinygpt.configs.config import ExperimentConfig, GPT2DataConfig

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

@torch.no_grad()
def evaluate(model: GPT2, valid_loader: DataLoader, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    for x, y in valid_loader:
        x = x.to(device)
        y = y.to(device)
        _, loss = model(idx=x, targets=y) # this loss is the average negative log-likelihood(per sample in the batch)
        batch_size = x.size(0)
        total_loss += batch_size * loss.item()
        total_samples += batch_size
    return total_loss / max(1, total_samples) # in case of 0 sample

def main():
    # ----------------------
    # 1. Setup Device and DDP
    # ----------------------
    # DDP Initialization (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    device, ddp_rank, ddp_local_rank, ddp_world_size, master_process = init_ddp(ddp)
    if device.startswith("cuda"):
        device_type = "cuda"

    # 2. Setup Model parameters and initialized tracking with SwanLab
    # ----------------------
    # todo: add resume checkpoints
    config = GPT2DataConfig(vocab_size=50304, batch_size=16, learning_rate=6e-4, device=device,
                            num_workers=16*2)
    config_dict = dataclasses.asdict(config)

    # todo: ddp_rank or ddp_local_rank?
    logger = setup_logger(cfg = config, train_name = "gpt2-shakespeare-v2-DDP", local_rank = ddp_local_rank)

    # All ranks (GPUs) must use the same seed to avoid non-determinism in DDP
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Initialize SwanLab
    swanlab.init(
        project="gpt2-training",  # Your project name
        experiment_name="gpt2-shakespeare-v2-DDP",
        config=config_dict,  # Log hyperparameters
        mode="local",  # Use local mode (no cloud sync)
        description = "GPT-2 124M experiment training on Shakespeare text",
        tags = ["GPT2", "Experiment", "Shakespeare", "small dataset"],
    )
    # TODO: test DDP master process verification process
    logger.info(f"I am GPU {ddp_rank}")
    # import sys; sys.exit(0)

    # ----------------------
    # 3. Setup Data(on CPU), Model, Optimizer
    # ----------------------
    # TODO: modify dataloader for DDP: process_rank=ddp_rank, num_processes=ddp_world_size
    # dl = make_dataloader(config)
    dl = make_ddp_dataloader(config)
    train_dl = dl["train_dl"]
    # valid_dl = dl["valid_dl"]
    test_dl = dl["test_dl"]

    # Applies TensorFloat32 operation only to CUDA matrix multiplication operations
    # TF32 is the precision-reduced variant of FP32, but not half-precision(FP16/BF16)
    # It is a specialized 19-bit floating-point format designed by NVIDIA for accelerating AI workloads, distinct from both 32-bit single-precision (FP32) and 16-bit half-precision (FP16/BF16) formats
    torch.set_float32_matmul_precision('high')

    # Create the model and move model to the GPU device
    model = GPT2(config)
    model.to(device)

    # Compile model speed up training, compilation itself consume some time
    # Efficiency gain comes from reducing Python overhead and GPU R/W
    ## note: compile is default, unless you are debugging and want to save compile time
    # Due to incompatibility of torch.compile() for MPS devices, only allow it for GPU devices.
    if device=="cuda":
        model = torch.compile(model)

    # Gradient Accumulations to fit GPT3 paper - GPT3 125M parameter model batch size 0.5M
    total_batch_size = 524288  # 2**19, closest with 0.5M
    assert total_batch_size % config.batch_size * config.block_size == 0
    # todo: gradient_accumulation_steps = total_batch_size / (config.batch_size * config.block_size) # 32
    # todo: delete after test
    gradient_accumulation_steps = 2
    if master_process:
        logger.info(f"total desired batch size: {total_batch_size}")
        logger.info(f"=> calculated gradient accumulation steps: {gradient_accumulation_steps}")

    # Number of steps to go through 1 epoch
    max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
    warmup_steps = 715

    # Optimizer sits outside the iteration loop
    # start with the max lr=6e-4
    optimizer = model.configure_optimizers(weight_decay=0.1, lr=model.config.learning_rate, device=device)

    # learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=model.config.learning_rate * 0.1)

    # ----------------------
    # 4. Training Tracking Variables and load from resume state
    # ----------------------
    best_valid_loss = float('inf')
    start_time = time.time()  # Track total training time

    # Load checkpoint (resume state)
    resume = False
    if resume:
        resume_shard_idx, resume_seq_idx, start_epoch, start_loss = load_training_checkpoint(
            cfg=config, model=model, optimizer=optimizer, scheduler=scheduler
        )
        # TODO: how to let train set resume from logged idx
        # train_dataset.resume_shard_idx = resume_shard_idx
        # train_dataset.resume_seq_idx = resume_seq_idx

    # ----------------------
    # 5. Training Loop
    # ----------------------
    log_every = model.config.eval_interval
    train_iter = iter(train_dl) # use a persistent iterator
    model.train() # todo: what does .train() do?
    # TODO: change steps with max_step in PROD
    max_steps = 2 # Experiment time spend: 1M tokens: train time: ; 1 eval run time:
    for i in range(max_steps):
        t0 = time.time() # current time in seconds since the Unix epoch

        # Forward and Backward Path
        # !! start with zero gradients
        optimizer.zero_grad()
        # Gradient accumulation to accumulate 0.5M token's of gradients every update step, remember to normalize the loss by grad_accum_steps due to "sum of average != overall average"
        loss_accum = 0.0
        for micro_step in range(gradient_accumulation_steps):
            # Load batch
            # x,y = next(iter(train_dl)) # Problem: You're using next(iter(train_dl)) inside the training loop, which re-initializes the DataLoader iterator every iteration. This is extremely inefficient and likely causing a CPU bottleneck that masks GPU speedups.
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
            loss_accum += loss.detach()

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

        # Log training time and metrics
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

        # if i == 0 or i % log_every == 9:
        #     eval_loss = evaluate(model, valid_dl, device)
        #     elapsed_time = time.time() - start_time  # Time since start in seconds
        #     # Update the best validation loss
        #     if eval_loss < best_valid_loss:
        #         best_valid_loss = eval_loss
        #     # Log train and validation loss
        #     swanlab.log({
        #         "Loss/valid_loss": eval_loss,
        #         "Loss/best_valid_loss": best_valid_loss,
        #     }, step=i)

        # Learning Rate scheduler update (happens after optimizer step)
        scheduler.step()

    # ----------------------
    # 6. Save model checkpoints per epoch
    # To pause/resume training:
    # * Save data progress (e.g., last processed shard + batch index) alongside model checkpoints.
    # * Save model weights, optimizer state, scheduler state, and data state (shard/batch).
    # ----------------------
    # save_training_checkpoint(cfg=config, model=model, optimizer=optimizer, scheduler=scheduler)
    # todo: bug: save_training_checkpoint missing 4 required positional arguments: 'shard_idx', 'seq_idx', 'epoch', and 'loss'
    # todo: save data config states

    # ----------------------
    # 7. Final Logging
    # ----------------------
    total_training_time = time.time() - start_time
    swanlab.log({"total_training_time": total_training_time})
    print(f"Training finished! Total time: {total_training_time:.2f}s")


if __name__ == '__main__':
    freeze_support()
    if sys.platform == 'darwin':
        torch.multiprocessing.set_start_method('spawn', force=True)
    main()
    # todo: add shutdown