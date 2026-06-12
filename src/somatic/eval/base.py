"""Base classes and protocols for evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor

if TYPE_CHECKING:
    from transformers.modeling_outputs import MaskedLMOutput


@runtime_checkable
class Metric(Protocol):
    """Protocol defining the interface for evaluation metrics.

    All metrics must implement this protocol to be used with the evaluation system.

    Class Attributes:
        name: Unique identifier for the metric (used in logging).
        requires_coords: Whether this metric requires coordinate data.
        needs_attentions: Whether this metric needs attention weights.
        needs_hidden_states: Whether this metric needs per-layer hidden states.
    """

    name: ClassVar[str]
    requires_coords: ClassVar[bool]
    needs_attentions: ClassVar[bool]
    needs_hidden_states: ClassVar[bool]

    def update(
        self,
        outputs: MaskedLMOutput,
        batch: dict[str, Tensor | None],
        labels: Tensor,
    ) -> None:
        """Accumulate metric values from a single batch.

        Args:
            outputs: Model outputs containing:
                - ``logits``: Output logits (batch, seq_len, vocab_size)
                - ``hidden_states``: (optional) per-layer hidden-state tuple
                  (requested via ``needs_hidden_states``); final layer is ``[-1]``
                - ``attentions``: (optional) tuple of attention tensors per layer
            batch: Input batch dictionary containing:
                - "input_ids": Original token IDs (batch, seq_len)
                - "token_type_ids": Chain identity (batch, seq_len)
                - "attention_mask": Padding mask (batch, seq_len)
                - "coords": (optional) 3D coordinates (batch, seq_len, 3)
            labels: MLM labels (batch, seq_len) — original ids at masked positions,
                ``-100`` elsewhere. The masked positions are ``labels != -100``.
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

    def state_objects(self) -> dict[str, Any] | list[Any] | None:
        """Return variable-length state as Python objects, or None.

        Metrics whose state cannot use tensor-based gathering (e.g. probes that
        accumulate variable numbers of feature vectors) return a non-None value
        here, which routes gathering through ``gather_object``.
        """
        ...

    def load_state_objects(self, gathered: list[Any]) -> None:
        """Load state from gathered Python objects (see ``state_objects``)."""
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
    needs_hidden_states: ClassVar[bool] = False

    def __init__(self) -> None:
        """Initialize the metric with default accumulators."""
        self._total: float = 0.0
        self._count: int = 0

    @abstractmethod
    def update(
        self,
        outputs: MaskedLMOutput,
        batch: dict[str, Tensor | None],
        labels: Tensor,
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

    def state_objects(self) -> dict[str, Any] | list[Any] | None:
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

    # Intentional no-op default: only metrics with object-based state override this.
    def load_state_objects(self, gathered: list[Any]) -> None:  # noqa: B027
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
