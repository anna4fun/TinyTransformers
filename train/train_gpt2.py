import torch
from torch.multiprocessing import freeze_support
import sys
from tinygpt.models.gpt2 import GPT2
from tinygpt.data_loaders.gpt2_data_loader import make_dataloader
from tinygpt.configs.config import ExperimentConfig


# Logging
# wandb.init(project="my-gpt2")
# logging.getLogger("lm_dataset").addHandler(wandb.logging.WandbHandler())

def train():
    # Devices
    device="cpu"
    if torch.cuda.is_available():
        device="cuda"
    elif torch.backends.mps.is_available():
        device="mps"
    print(device)

    toyconfig = ExperimentConfig
    # dl = make_dataloader(GPT2DataConfig)
    dl = make_dataloader(toyconfig)
    train_dl = dl["train_dl"]
    valid_dl = dl["valid_dl"]
    type(train_dl)

    # take one sample from train_dl
    x, y = next(iter(train_dl))
    print(x.shape)
    print(y.shape)
    # model = GPT2(GPT2DataConfig)
    model = GPT2(toyconfig)
    print("model device:", next(model.parameters()).device)
    print("x device:", x.device)
    print("y device:", y.device)
    # move model, x, y to mps
    x = x.to(device)
    y = y.to(device)
    model.to(device)

    with torch.no_grad():
        logits, loss = model.forward(x, y)
    print(loss)

if __name__ == '__main__':
    freeze_support()
    if sys.platform == 'darwin':
        torch.multiprocessing.set_start_method('spawn', force=True)
    train()