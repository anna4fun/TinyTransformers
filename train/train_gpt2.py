import wandb
import logging
from gpt2 import GPT2
from data_loaders import gpt2_data_loader

# Logging
wandb.init(project="my-gpt2")
logging.getLogger("lm_dataset").addHandler(wandb.logging.WandbHandler())
