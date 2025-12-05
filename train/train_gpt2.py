import dataclasses
import time

import torch
from torch.multiprocessing import freeze_support
import sys
import swanlab
import os
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
    print(device)

    # ----------------------
    # 2. Setup Model parameters and initialized tracking with SwanLab
    # ----------------------
    toyconfig = GPT2DataConfig()
    config_dict = dataclasses.asdict(toyconfig)
    # Initialize SwanLab
    swanlab.init(
        project="gpt2-training",  # Your project name
        experiment_name="gpt2-shakespeare-v1-wte-lm_head-share-weights",
        config=config_dict,  # Log hyperparameters
        mode="local",  # Use local mode (no cloud sync)
        description = "GPT-2 124M experiment training on Shakespeare text (custom learning rate)",
        tags = ["GPT2", "Experiment", "Shakespeare", "small dataset"],
    )

    # ----------------------
    # 3. Setup Data(on CPU), Model, Optimizer
    # ----------------------
    dl = make_dataloader(toyconfig)
    train_dl = dl["train_dl"]
    valid_dl = dl["valid_dl"]
    # create the model and move model, x, y to the GPU device
    model = GPT2(toyconfig)
    model.to(device)
    # interesting: optimizer sits outside the iteration loop
    # AdamW fixes the bug of Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=toyconfig.learning_rate)

    # ----------------------
    # 4. Training Tracking Variables
    # ----------------------
    best_valid_loss = float('inf')
    start_time = time.time()  # Track total training time

    # ----------------------
    # 5. Training Loop
    # ----------------------
    steps = 50
    log_every = 10

    model.train() # todo: what does .train() do?
    for i in range(steps):
        x,y = next(iter(train_dl))
        x = x.to(device)
        y = y.to(device)
        # Forward and Backward Path
        # !! start with zero gradients
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if i % log_every == 0:
            eval_loss = evaluate(model, valid_dl, device)
            elapsed_time = time.time() - start_time  # Time since start
            # Update the best validation loss
            if eval_loss < best_valid_loss:
                best_valid_loss = eval_loss
            # Log to SwanLab
            swanlab.log({
                "Loss/train_loss": loss.item(),
                "Loss/valid_loss": eval_loss,
                "Loss/best_valid_loss": best_valid_loss,
                "elapsed_time": elapsed_time,
                "learning_rate": toyconfig.learning_rate
            }, step=i)  # Step = current iteration

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