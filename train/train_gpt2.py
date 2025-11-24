import torch
import wandb
import logging
from gpt2 import GPT2
from data_loaders.gpt2_data_loader import make_dataloader
from config import GPT2DataConfig

# Logging
# wandb.init(project="my-gpt2")
# logging.getLogger("lm_dataset").addHandler(wandb.logging.WandbHandler())

# Devices
device="cpu"
if torch.cuda.is_available():
    device="cuda"
elif torch.backends.mps.is_available():
    device="mps"
print(device)

dl = make_dataloader(GPT2DataConfig)
train_dl = dl["train_dl"]
valid_dl = dl["valid_dl"]
type(train_dl)

# take one sample from train_dl
x, y = next(iter(train_dl))
print(x.shape)
print(y.shape)
model = GPT2(GPT2DataConfig)

print("model device:", next(model.parameters()).device)
print("x device:", x.device)
print("y device:", y.device)
# move model, x, y to mps
x = x.to(device)
y = y.to(device)
model.to(device)

with torch.no_grad():
    logits, loss = model.forward(x, y)
# Process finished with exit code 137 (interrupted by signal 9:SIGKILL)
print(loss)