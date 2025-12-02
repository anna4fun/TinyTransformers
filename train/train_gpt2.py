import dataclasses
import torch
from torch.multiprocessing import freeze_support
import sys
import swanlab
from torch.utils.data import DataLoader
from tinygpt.models.gpt2 import GPT2
from tinygpt.data_loaders.gpt2_data_loader import make_dataloader
from tinygpt.configs.config import ExperimentConfig

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
    # Devices
    device="cpu"
    if torch.cuda.is_available():
        device="cuda"
    elif torch.backends.mps.is_available():
        device="mps"
    print(device)

    toyconfig = ExperimentConfig()
    config_dict = dataclasses.asdict(toyconfig)
    dl = make_dataloader(toyconfig)
    train_dl = dl["train_dl"]
    valid_dl = dl["valid_dl"]
    type(train_dl)

    # take one sample from train_dl
    x, y = next(iter(train_dl))
    print(x.shape)
    print(y.shape)
    # create the model and move model, x, y to the GPU device
    model = GPT2(toyconfig)
    x = x.to(device)
    y = y.to(device)
    model.to(device)

    # interesting: optimizer sits outside the iteration loop
    # AdamW fixes the bug of Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=toyconfig.learning_rate)

    # training loop
    steps = 50
    log_every = 10

    model.train() # todo: what does .train() do?
    for i in range(steps):
        _, loss = model.forward(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % log_every == 0:
            eval_loss = evaluate(model, valid_dl, device)
            print(f"Step {i}: Train Loss:{loss.item():.3f},  Eval Loss: {eval_loss:.3f}")


if __name__ == '__main__':
    freeze_support()
    if sys.platform == 'darwin':
        torch.multiprocessing.set_start_method('spawn', force=True)
    main()