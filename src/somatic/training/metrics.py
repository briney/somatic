"""Training and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class MetricAccumulator:
    """Accumulates metrics over steps for averaging."""

    _values: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def update(self, name: str, value: float, count: int = 1) -> None:
        if name not in self._values:
            self._values[name] = 0.0
            self._counts[name] = 0

        self._values[name] += value * count
        self._counts[name] += count

    def compute(self, name: str) -> float | None:
        if name not in self._values or self._counts[name] == 0:
            return None
        return self._values[name] / self._counts[name]

    def compute_all(self) -> dict[str, float]:
        return {name: self.compute(name) for name in self._values}

    def reset(self) -> None:
        self._values.clear()
        self._counts.clear()


def compute_masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask_labels: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Compute cross-entropy loss only on masked positions."""
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    mask_flat = mask_labels.view(-1)

    loss_per_token = torch.nn.functional.cross_entropy(
        logits_flat, targets_flat, reduction="none"
    )

    masked_loss = loss_per_token * mask_flat.float()

    if reduction == "none":
        return masked_loss.view(batch_size, seq_len)
    elif reduction == "sum":
        return masked_loss.sum()
    else:
        num_masked = mask_flat.sum().clamp(min=1)
        return masked_loss.sum() / num_masked


def compute_accuracy(
    logits: Tensor, targets: Tensor, mask_labels: Tensor
) -> tuple[float, int]:
    """Compute accuracy on masked positions."""
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets) & mask_labels.bool()

    num_correct = correct.sum().item()
    num_total = mask_labels.sum().item()

    if num_total == 0:
        return 0.0, 0

    return num_correct / num_total, num_total


def compute_perplexity(loss: Tensor) -> Tensor:
    """Compute perplexity from loss."""
    return torch.exp(loss)


@dataclass
class MLMMetrics:
    """Container for MLM training metrics."""

    loss: float
    accuracy: float
    perplexity: float
    num_masked_tokens: int
    mask_rate: float

    def to_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "perplexity": self.perplexity,
            "num_masked_tokens": self.num_masked_tokens,
            "mask_rate": self.mask_rate,
        }


def compute_mlm_metrics(
    logits: Tensor,
    targets: Tensor,
    mask_labels: Tensor,
    attention_mask: Tensor,
) -> MLMMetrics:
    """Compute all MLM training metrics."""
    loss = compute_masked_cross_entropy(logits, targets, mask_labels)
    accuracy, num_masked = compute_accuracy(logits, targets, mask_labels)
    perplexity = compute_perplexity(loss).item()

    valid_tokens = attention_mask.sum().item()
    mask_rate = num_masked / valid_tokens if valid_tokens > 0 else 0.0

    return MLMMetrics(
        loss=loss.item(),
        accuracy=accuracy,
        perplexity=perplexity,
        num_masked_tokens=num_masked,
        mask_rate=mask_rate,
    )
