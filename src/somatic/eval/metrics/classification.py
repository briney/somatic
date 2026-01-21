"""Classification metrics for masked language modeling evaluation."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor

from ..base import MetricBase
from ..registry import register_metric


@register_metric("masked_accuracy")
class MaskedAccuracyMetric(MetricBase):
    """Masked token accuracy for MLM pre-training.

    Computes the fraction of correctly predicted masked tokens, ignoring
    positions marked with the ignore_index or not masked.
    """

    name: ClassVar[str] = "mask_acc"
    requires_coords: ClassVar[bool] = False
    needs_attentions: ClassVar[bool] = False

    def __init__(self, **kwargs) -> None:
        """Initialize the metric."""
        super().__init__()
        self._correct: int = 0
        self._total: int = 0

    def update(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate accuracy from a batch.

        Args:
            outputs: Model outputs with "logits" key.
            batch: Input batch with "token_ids" (original tokens).
            mask_labels: Binary mask indicating masked positions.
        """
        logits = outputs["logits"]
        targets = batch["token_ids"]

        # Get predictions
        predictions = logits.argmax(dim=-1)

        # Only evaluate masked positions
        mask = mask_labels.bool()
        correct = (predictions == targets) & mask

        self._correct += correct.sum().item()
        self._total += mask.sum().item()

    def compute(self) -> dict[str, float]:
        """Compute accuracy from accumulated counts.

        Returns:
            Dictionary with "mask_acc" key.
        """
        if self._total == 0:
            return {self.name: 0.0}
        return {self.name: self._correct / self._total}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._correct = 0
        self._total = 0

    def state_tensors(self) -> list[Tensor]:
        """Return state as tensors for distributed aggregation."""
        return [torch.tensor([float(self._correct), float(self._total)])]

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors and len(tensors) > 0:
            state = tensors[0]
            self._correct = int(state[0].item())
            self._total = int(state[1].item())


@register_metric("perplexity")
class PerplexityMetric(MetricBase):
    """Perplexity metric computed as exp(cross-entropy loss).

    Measures how well the model predicts masked tokens. Lower is better.
    """

    name: ClassVar[str] = "ppl"
    requires_coords: ClassVar[bool] = False
    needs_attentions: ClassVar[bool] = False

    def __init__(self, **kwargs) -> None:
        """Initialize the metric."""
        super().__init__()
        self._total_loss: float = 0.0
        self._total_tokens: int = 0

    def update(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate loss for perplexity computation.

        Args:
            outputs: Model outputs with "logits" key.
            batch: Input batch with "token_ids" (original tokens).
            mask_labels: Binary mask indicating masked positions.
        """
        logits = outputs["logits"]
        targets = batch["token_ids"]

        # Compute cross-entropy loss on masked positions
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        mask_flat = mask_labels.view(-1).bool()

        # Get per-token loss
        loss_per_token = torch.nn.functional.cross_entropy(
            logits_flat, targets_flat, reduction="none"
        )

        # Sum loss only on masked positions
        masked_loss = loss_per_token[mask_flat].sum()
        num_masked = mask_flat.sum()

        self._total_loss += masked_loss.item()
        self._total_tokens += num_masked.item()

    def compute(self) -> dict[str, float]:
        """Compute perplexity from accumulated loss.

        Returns:
            Dictionary with "ppl" key.
        """
        if self._total_tokens == 0:
            return {self.name: float("inf")}

        avg_loss = self._total_loss / self._total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return {self.name: perplexity}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._total_loss = 0.0
        self._total_tokens = 0

    def state_tensors(self) -> list[Tensor]:
        """Return state as tensors for distributed aggregation."""
        return [torch.tensor([self._total_loss, float(self._total_tokens)])]

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors and len(tensors) > 0:
            state = tensors[0]
            self._total_loss = state[0].item()
            self._total_tokens = int(state[1].item())


@register_metric("loss")
class LossMetric(MetricBase):
    """Average cross-entropy loss on masked tokens.

    This is the raw loss value without exponentiating.
    """

    name: ClassVar[str] = "loss"
    requires_coords: ClassVar[bool] = False
    needs_attentions: ClassVar[bool] = False

    def __init__(self, **kwargs) -> None:
        """Initialize the metric."""
        super().__init__()
        self._total_loss: float = 0.0
        self._total_tokens: int = 0

    def update(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate loss from a batch.

        Args:
            outputs: Model outputs with "logits" key.
            batch: Input batch with "token_ids" (original tokens).
            mask_labels: Binary mask indicating masked positions.
        """
        logits = outputs["logits"]
        targets = batch["token_ids"]

        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        mask_flat = mask_labels.view(-1).bool()

        loss_per_token = torch.nn.functional.cross_entropy(
            logits_flat, targets_flat, reduction="none"
        )

        masked_loss = loss_per_token[mask_flat].sum()
        num_masked = mask_flat.sum()

        self._total_loss += masked_loss.item()
        self._total_tokens += num_masked.item()

    def compute(self) -> dict[str, float]:
        """Compute average loss.

        Returns:
            Dictionary with "loss" key.
        """
        if self._total_tokens == 0:
            return {self.name: float("inf")}
        return {self.name: self._total_loss / self._total_tokens}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._total_loss = 0.0
        self._total_tokens = 0

    def state_tensors(self) -> list[Tensor]:
        """Return state as tensors for distributed aggregation."""
        return [torch.tensor([self._total_loss, float(self._total_tokens)])]

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors and len(tensors) > 0:
            state = tensors[0]
            self._total_loss = state[0].item()
            self._total_tokens = int(state[1].item())
