"""Base classes and protocols for evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class Metric(Protocol):
    """Protocol defining the interface for evaluation metrics.

    All metrics must implement this protocol to be used with the evaluation system.

    Class Attributes:
        name: Unique identifier for the metric (used in logging).
        requires_coords: Whether this metric requires coordinate data.
        needs_attentions: Whether this metric needs attention weights.
    """

    name: ClassVar[str]
    requires_coords: ClassVar[bool]
    needs_attentions: ClassVar[bool]

    def update(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate metric values from a single batch.

        Args:
            outputs: Model outputs dictionary containing:
                - "logits": Output logits (batch, seq_len, vocab_size)
                - "hidden_states": Final hidden states (batch, seq_len, d_model)
                - "attentions": (optional) Tuple of attention tensors per layer
            batch: Input batch dictionary containing:
                - "token_ids": Original token IDs (batch, seq_len)
                - "chain_ids": Chain identity (batch, seq_len)
                - "attention_mask": Padding mask (batch, seq_len)
                - "coords": (optional) 3D coordinates (batch, seq_len, 3)
            mask_labels: Mask indicating which tokens were masked (batch, seq_len).
        """
        ...

    def compute(self) -> dict[str, float]:
        """Compute final metric value(s) from accumulated state.

        Returns:
            Dictionary mapping metric names to float values.
        """
        ...

    def reset(self) -> None:
        """Reset accumulated state for a new evaluation run."""
        ...

    def state_tensors(self) -> list[Tensor]:
        """Return internal state as tensors for distributed aggregation.

        Returns:
            List of tensors representing the metric's accumulated state.
        """
        ...

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        """Restore state from gathered tensors (for distributed training).

        Args:
            tensors: List of tensors as returned by state_tensors(),
                potentially aggregated across processes.
        """
        ...


class MetricBase(ABC):
    """Abstract base class for metrics with default implementations.

    Provides default implementations for state_tensors() and load_state_tensors()
    that work for simple scalar accumulators. Subclasses should override these
    if they have more complex state.

    Class Attributes:
        name: Unique identifier for the metric.
        requires_coords: Whether this metric requires coordinate data.
        needs_attentions: Whether this metric needs attention weights.
    """

    name: ClassVar[str] = ""
    requires_coords: ClassVar[bool] = False
    needs_attentions: ClassVar[bool] = False

    def __init__(self) -> None:
        """Initialize the metric with default accumulators."""
        self._total: float = 0.0
        self._count: int = 0

    @abstractmethod
    def update(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate metric values from a single batch."""
        ...

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Compute final metric value(s) from accumulated state."""
        ...

    def reset(self) -> None:
        """Reset accumulated state for a new evaluation run."""
        self._total = 0.0
        self._count = 0

    def state_tensors(self) -> list[Tensor]:
        """Return internal state as tensors for distributed aggregation.

        Default implementation returns [total, count] as a single tensor.
        Override for metrics with more complex state.
        """
        return [torch.tensor([self._total, float(self._count)])]

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        """Restore state from gathered tensors.

        Default implementation expects the format from state_tensors().
        Override for metrics with more complex state.
        """
        if tensors and len(tensors) > 0:
            state = tensors[0]
            self._total = state[0].item()
            self._count = int(state[1].item())

    def state_objects(self) -> list[Any] | None:
        """Return state as Python objects for distributed gathering.

        Used for metrics with variable-length state that cannot use
        tensor-based gathering (e.g., lists of different sizes per process).
        When this returns a non-None value, the evaluator will use
        accelerator.gather_object() instead of tensor gathering.

        Default implementation returns None, meaning tensor-based gathering
        should be used. Override this for metrics with variable-length state.

        Returns:
            List of Python objects to gather, or None to use tensor gathering.
        """
        return None

    def load_state_objects(self, gathered: list[Any]) -> None:
        """Load state from gathered Python objects.

        Called after gather_object collects data from all processes.
        The gathered argument is a list containing state_objects() results
        from each process.

        Default implementation does nothing. Override this if the metric
        uses object-based gathering.

        Args:
            gathered: List of objects gathered from all processes.
        """
        pass
