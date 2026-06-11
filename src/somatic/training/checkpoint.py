"""Checkpointing utilities for training.

Checkpoints are HuggingFace model directories (clean break from the legacy
combined ``.pt`` format). Each checkpoint directory holds:

- ``config.json`` + ``model.safetensors`` (via ``model.save_pretrained``), with
  ``auto_map`` and the bundled custom-code files for ``trust_remote_code`` loading;
- ``tokenizer.json`` + ``tokenizer_config.json`` (via ``tokenizer.save_pretrained``);
- ``training_state.pt`` — the resume state (optimizer, scheduler, RNG, step/epoch,
  metrics).

Resume rebuilds the model with ``AutoModelForMaskedLM.from_pretrained(dir)`` and
then restores the training state via :meth:`CheckpointManager.load_training_state`.
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..model.tokenization_somatic import tokenizer

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from ..model import SomaticForMaskedLM


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
    """Saves and loads HuggingFace-format training checkpoints."""

    def __init__(
        self,
        config: CheckpointConfig,
        model: SomaticForMaskedLM,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric_value: float | None = None
        self.saved_checkpoints: list[Path] = []

    def _is_better(self, current: float, best: float) -> bool:
        if self.config.best_mode == "min":
            return current < best
        return current > best

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        """Snapshot Python/NumPy/Torch (and CUDA) RNG so resume is reproducible."""
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict[str, Any] | None) -> None:
        if not state:
            return
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def _write(
        self,
        ckpt_dir: Path,
        step: int,
        epoch: float,
        metrics: dict[str, float] | None,
    ) -> None:
        """Write a complete checkpoint directory (model + tokenizer + train state)."""
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        training_state: dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "rng": self._capture_rng_state(),
            "metrics": metrics or {},
        }
        torch.save(training_state, ckpt_dir / "training_state.pt")

    def save(
        self,
        step: int,
        epoch: float,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save a checkpoint directory, rotating old ones and tracking the best."""
        ckpt_dir = self.save_dir / f"checkpoint_step_{step}"
        self._write(ckpt_dir, step, epoch, metrics)
        # Move ckpt_dir to the most-recent slot (deduped) so re-saving the same
        # step — e.g. a final checkpoint that lands on a checkpoint_steps boundary —
        # never schedules the directory we just wrote for deletion.
        self.saved_checkpoints = [p for p in self.saved_checkpoints if p != ckpt_dir]
        self.saved_checkpoints.append(ckpt_dir)

        # Rotate out the oldest checkpoints beyond keep_last_n.
        while len(self.saved_checkpoints) > self.config.keep_last_n:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint != ckpt_dir and old_checkpoint.exists():
                shutil.rmtree(old_checkpoint)

        # Track the best checkpoint by the configured metric.
        if self.config.save_best and metrics is not None:
            metric_value = metrics.get(self.config.best_metric)
            if metric_value is not None and (
                self.best_metric_value is None
                or self._is_better(metric_value, self.best_metric_value)
            ):
                self.best_metric_value = metric_value
                best_dir = self.save_dir / "best_checkpoint"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                self._write(best_dir, step, epoch, metrics)

        return ckpt_dir

    def _resolve_dir(self, checkpoint_path: str | None, load_best: bool) -> Path:
        if load_best:
            return self.save_dir / "best_checkpoint"
        if checkpoint_path is not None:
            return Path(checkpoint_path)
        # Latest checkpoint by step number (not lexicographic order).
        checkpoints = list(self.save_dir.glob("checkpoint_step_*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.save_dir}")
        return max(checkpoints, key=lambda p: int(p.name.rsplit("_", 1)[-1]))

    def load_training_state(
        self,
        checkpoint_path: str | None = None,
        load_best: bool = False,
        map_location: str = "cpu",
    ) -> dict[str, Any]:
        """Restore optimizer/scheduler/RNG from a checkpoint's ``training_state.pt``.

        The model weights are NOT loaded here — rebuild the model with
        ``SomaticForMaskedLM.from_pretrained(dir)`` before constructing the
        optimizer this manager wraps.

        Returns:
            ``{"step", "epoch", "metrics"}`` from the saved state.
        """
        path = self._resolve_dir(checkpoint_path, load_best)
        state = torch.load(
            path / "training_state.pt", map_location=map_location, weights_only=False
        )

        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and state.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self._restore_rng_state(state.get("rng"))

        return {
            "step": state["step"],
            "epoch": state["epoch"],
            "metrics": state.get("metrics", {}),
        }

    def should_save(self, step: int) -> bool:
        """Check if we should save at this step."""
        return step > 0 and step % self.config.checkpoint_steps == 0
