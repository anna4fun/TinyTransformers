import math
import torch
from tinygpt.models.gpt2 import GPT2
from torch.utils.data import DataLoader


@torch.no_grad()
def loss_evaluate(model: GPT2, valid_loader: DataLoader, device, max_val_batches=100, perplexity_required = False):
    # Total of 1000,000 seq in val_dl, pick 100 for validation every time, validate for 40 times
    # Switch to eval mode, turns off layers like dropout/norms so the model behave deterministicaly.
    # Without eval mode, the loss would vary batch by batch and the metric would be noisy.
    model.eval()
    
    total_loss, total_samples = 0.0, 0
    val_loss, val_perplexity = 0.0, 0.0
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
        if perplexity_required:
            val_perplexity = math.exp(val_loss)

        # Re-enable the stochastic behavior (eg. dropout) needed for learning. 
        # If missing, the next train batch would be in inference mode and cause divergence or stall
        model.train()
    return val_loss, val_perplexity # in case of 0 sample