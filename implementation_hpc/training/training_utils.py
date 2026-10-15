import math
from typing import Dict, List

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """Cosine learning rate scheduler with initial linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        if step < self.warmup_steps:
            alpha = step / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


def save_checkpoint(
    state_dict: Dict,
    epoch: int,
    val_loss: float,
    val_recall: float,
    checkpoint_path: str,
) -> None:
    """Saves training checkpoint including optimizer state and performance metrics."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "val_loss": val_loss,
            "val_recall": val_recall,
        },
        checkpoint_path,
    )
