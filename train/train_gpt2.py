import code
import dataclasses
import time

import torch
from torch.multiprocessing import freeze_support
import sys
import swanlab
import os

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tinygpt.models.gpt2 import GPT2
from tinygpt.data_loaders.gpt2_data_loader import make_dataloader
from tinygpt.configs.config import ExperimentConfig, GPT2DataConfig

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
    # 1. Setup Device
    # ----------------------
    device="cpu"
    if torch.cuda.is_available():
        device="cuda"
    elif torch.backends.mps.is_available():
        device="mps"

    # ----------------------
    # 2. Setup Model parameters and initialized tracking with SwanLab
    # ----------------------
    # TODO: change the B and T into original size 64 and 1024
    toyconfig = GPT2DataConfig(vocab_size=50304, batch_size=4, block_size=32, learning_rate = 6e-4)
    config_dict = dataclasses.asdict(toyconfig)
    config_dict["device"] = device

    # Initialize SwanLab
    swanlab.init(
        project="gpt2-training",  # Your project name
        experiment_name="gpt2-shakespeare-v1-weight-decay",
        config=config_dict,  # Log hyperparameters
        mode="local",  # Use local mode (no cloud sync)
        description = "GPT-2 124M experiment training on Shakespeare text",
        tags = ["GPT2", "Experiment", "Shakespeare", "small dataset"],
    )
    # TODO: test DDP master process verification process
    # Logging("I am GPU", ddp_rank)
    # import sys; sys.exit(0)

    # ----------------------
    # 3. Setup Data(on CPU), Model, Optimizer
    # ----------------------
    # TODO: modify dataloader for DDP: process_rank=ddp_rank, num_processes=ddp_world_size
    dl = make_dataloader(toyconfig)
    train_dl = dl["train_dl"]
    valid_dl = dl["valid_dl"]
    # Applies TensorFloat32 operation only to CUDA matrix multiplication operations
    # TF32 is the precision-reduced variant of FP32, but not half-precision(FP16/BF16)
    # It is a specialized 19-bit floating-point format designed by NVIDIA for accelerating AI workloads, distinct from both 32-bit single-precision (FP32) and 16-bit half-precision (FP16/BF16) formats
    torch.set_float32_matmul_precision('high')
    # create the model and move model, x, y to the GPU device
    model = GPT2(toyconfig)
    model.to(device)
    # Compile model speed up training, compilation itself consume some time
    # Efficiency gain comes from reducing Python overhead and GPU R/W
    ## note: compile is default, unless you are debugging and want to save compile time
    # Due to incompatibility of torch.compile() for MPS devices, only allow it for GPU devices.
    if device=="cuda":
        t0 = time.time()
        model = torch.compile(model)
        compile_time = time.time() - t0
        swanlab.log({"model_compile_time": compile_time})
    # interesting: optimizer sits outside the iteration loop
    # AdamW fixes the bug of Adam
    optimizer = model.configure_optimizers(weight_decay=0.1, device=device)
    # learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # ----------------------
    # 4. Training Tracking Variables
    # ----------------------
    best_valid_loss = float('inf')
    start_time = time.time()  # Track total training time

    # ----------------------
    # 5. Training Loop
    # ----------------------
    steps = 20
    log_every = 10
    train_iter = iter(train_dl) # use a persistent iterator
    model.train() # todo: what does .train() do?
    for i in range(steps):
        t0 = time.time() # current time in seconds since the Unix epoch
        # Forward and Backward Path
        # !! start with zero gradients
        optimizer.zero_grad()
        # TODO: gradient accumulation, accumulate 0.5M token's of gradients every update step, remember to normalize the loss by grad_accum_steps due to "sum of average != overall average"
        # loss_accum = 0.0; loss_accum += loss.detach() # track leave nodes
        # for micro_step in range(grad_accum_steps):
        # Load batch
        # x,y = next(iter(train_dl)) # Problem: You're using next(iter(train_dl)) inside the training loop, which re-initializes the DataLoader iterator every iteration. This is extremely inefficient and likely causing a CPU bottleneck that masks GPU speedups.
        x, y = next(train_iter)
        x = x.to(device)
        y = y.to(device)
        # Autocast to BF16 in the forward path (BF16 only available on Ampere architecture GPUs)
        with torch.autocast(device_type = device,dtype=torch.bfloat16):
            logits, loss = model(x, y)
            # Pause training to inspect state, e.g. the precision of logits should show float16, while wte.weights are float32
            # code.interact(local=locals())
        # Exit autocast before the backward path

        # TODO: if ddp, only synchronize the gradients when it is the last step of the gradient accumulated batch
        # The backward path, get the raw gradients and deposit(sum) them into params.grad
        loss.backward()

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
        t1 = time.time()
        dt = t1- t0 # time difference in seconds
        # tokens per seconds
        B, T = x.shape
        # TODO: adjust token_per_sec with the gradient accumulated number of tokens 0.5M
        token_per_sec = (B*T)/dt
        # log running time and train loss every iteration
        swanlab.log({
            "Loss/train_loss": loss.item(),
            "Time/current_iteration_time": dt,
            "Time/token_per_sec": token_per_sec,
            "Norm": round(norm.item(), 6),
            "Learning Rate": float(scheduler.get_last_lr()[0]),
            # TODO: track intermediate loss_accum
        }, step=i)

        if i == 0 or i % log_every == 9:
            eval_loss = evaluate(model, valid_dl, device)
            elapsed_time = time.time() - start_time  # Time since start in seconds
            # Update the best validation loss
            if eval_loss < best_valid_loss:
                best_valid_loss = eval_loss
            # Log train and validation loss
            swanlab.log({
                "Loss/valid_loss": eval_loss,
                "Loss/best_valid_loss": best_valid_loss,
            }, step=i)
            # TODO: debug why the learning rate doesn't change in iter = 10 and 20
            scheduler.step()

    # ----------------------
    # 6. Final Logging
    # ----------------------
    total_training_time = time.time() - start_time
    swanlab.log({"total_training_time": total_training_time})
    print(f"Training finished! Total time: {total_training_time:.2f}s")


if __name__ == '__main__':
    freeze_support()
    if sys.platform == 'darwin':
        torch.multiprocessing.set_start_method('spawn', force=True)
    main()