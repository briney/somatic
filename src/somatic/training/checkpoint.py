"""Checkpointing utilities for training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""

    save_dir: str = "checkpoints"
    checkpoint_steps: int = 1000
    keep_last_n: int = 5
    save_best: bool = True
    best_metric: str = "val_loss"
    best_mode: str = "min"


class CheckpointManager:
    """Manages saving and loading of training checkpoints."""

    def __init__(
        self,
        config: CheckpointConfig,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler | None = None,
        model_config: Any | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_config = model_config

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric_value: float | None = None
        self.saved_checkpoints: list[Path] = []

    def _is_better(self, current: float, best: float) -> bool:
        if self.config.best_mode == "min":
            return current < best
        return current > best

    def save(
        self,
        step: int,
        epoch: int,
        metrics: dict[str, float] | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> Path | None:
        """Save a checkpoint."""
        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics or {},
        }

        if self.model_config is not None:
            checkpoint["config"] = asdict(self.model_config)

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        checkpoint_path = self.save_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.saved_checkpoints.append(checkpoint_path)

        # Remove old checkpoints
        while len(self.saved_checkpoints) > self.config.keep_last_n:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        # Save best checkpoint
        if self.config.save_best and metrics is not None:
            metric_value = metrics.get(self.config.best_metric)
            if metric_value is not None:
                if self.best_metric_value is None or self._is_better(
                    metric_value, self.best_metric_value
                ):
                    self.best_metric_value = metric_value
                    best_path = self.save_dir / "best_checkpoint.pt"
                    torch.save(checkpoint, best_path)

        return checkpoint_path

    def load(
        self,
        checkpoint_path: str | None = None,
        load_best: bool = False,
        map_location: str = "cpu",
    ) -> dict[str, Any]:
        """Load a checkpoint."""
        if load_best:
            path = self.save_dir / "best_checkpoint.pt"
        elif checkpoint_path is not None:
            path = Path(checkpoint_path)
        else:
            # Load latest checkpoint
            checkpoints = sorted(self.save_dir.glob("checkpoint_step_*.pt"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            path = checkpoints[-1]

        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "step": checkpoint["step"],
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint.get("metrics", {}),
            "extra_state": checkpoint.get("extra_state", {}),
        }

    def should_save(self, step: int) -> bool:
        """Check if we should save at this step."""
        return step > 0 and step % self.config.checkpoint_steps == 0
