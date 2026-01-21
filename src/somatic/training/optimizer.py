"""Optimizer and learning rate scheduler configuration."""

from __future__ import annotations

import math

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Optimizer:
    """Create AdamW optimizer with proper weight decay separation."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "bias" in name or "layer_norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(param_groups, lr=lr, betas=betas, eps=eps)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_decay: str = "cosine",
    num_training_steps: int = 100000,
    num_warmup_steps: int = 1000,
    min_lr_ratio: float = 0.1,
) -> _LRScheduler:
    """Create learning rate scheduler with warmup.

    Uses a single LambdaLR scheduler with explicit step-based multiplier calculation.
    This approach is more robust than SequentialLR when used with Accelerate.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_decay: Type of decay after warmup ("constant", "linear", or "cosine").
        num_training_steps: Total number of training steps.
        num_warmup_steps: Number of warmup steps.
        min_lr_ratio: Minimum learning rate as a ratio of the base learning rate.

    Returns:
        A LambdaLR scheduler.
    """
    if scheduler_decay not in ("constant", "linear", "cosine"):
        raise ValueError(f"Unknown scheduler decay type: {scheduler_decay}")

    # Pre-compute and capture as concrete Python values (not OmegaConf references)
    # This ensures the lambda captures actual ints/floats, not config objects
    warmup_steps = int(num_warmup_steps)
    total_steps = int(num_training_steps)
    decay_steps = max(0, total_steps - warmup_steps)
    min_lr = float(min_lr_ratio)
    decay_type = str(scheduler_decay).lower()

    def lr_lambda(current_step: int) -> float:
        # Warmup phase (0 -> 1)
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Constant - stay at 1.0
        if decay_type == "constant":
            return 1.0

        # Decay phase
        if decay_steps <= 0:
            return 1.0

        t = current_step - warmup_steps
        progress = min(max(float(t) / float(decay_steps), 0.0), 1.0)

        if decay_type == "cosine":
            # Cosine decay from 1.0 to min_lr
            return min_lr + 0.5 * (1.0 - min_lr) * (1.0 + math.cos(math.pi * progress))
        else:  # linear
            # Linear decay from 1.0 to min_lr
            return min_lr + (1.0 - min_lr) * (1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def get_lr(optimizer: Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]["lr"]
