import wandb
import logging

wandb.init(project="my-gpt2")

logging.getLogger("lm_dataset").addHandler(wandb.logging.WandbHandler())
