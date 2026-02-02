import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from tinygpt.evaluation_utils.eval_loss import loss_evaluate


class DummyLossModel:
    """Simple callable that returns a preset loss per batch and tracks training state."""

    def __init__(self, losses):
        self._losses = losses.copy()
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, x, y):
        return None, torch.tensor(self._losses.pop(0), dtype=torch.float32)


def test_loss_evaluate_accumulates_loss_and_perplexity():
    dataset = TensorDataset(torch.ones((5, 1)), torch.ones((5, 1)))
    loader = DataLoader(dataset, batch_size=2)
    model = DummyLossModel([0.5, 1.0, 0.25])

    val_loss, val_perplexity = loss_evaluate(
        model, loader, device="cpu", max_val_batches=3, perplexity_required=True
    )

    expected_loss = (2 * 0.5 + 2 * 1.0 + 1 * 0.25) / 5
    assert math.isclose(val_loss, expected_loss, rel_tol=1e-6)
    assert math.isclose(val_perplexity, math.exp(expected_loss), rel_tol=1e-6)
    assert model.training
