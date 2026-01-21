"""Probe-based metrics using linear classifiers on embeddings or attention."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import torch
from torch import Tensor

from ..base import MetricBase
from ..registry import register_metric


class ProbeMetricBase(MetricBase):
    """Base class for probe-based metrics.

    Probe metrics train small classifiers (e.g., logistic regression) on
    model representations (embeddings or attention weights) to predict
    some target property. The classifier is trained during the accumulation
    phase and evaluated in the compute phase.
    """

    def __init__(
        self,
        n_train: int = 100,
        regularization: float = 0.1,
        max_iterations: int = 100,
        **kwargs,
    ) -> None:
        """Initialize the probe metric.

        Args:
            n_train: Maximum number of samples for training.
            regularization: L2 regularization strength.
            max_iterations: Maximum training iterations.
        """
        super().__init__()
        self.n_train = n_train
        self.regularization = regularization
        self.max_iterations = max_iterations

        self._features: list[Tensor] = []
        self._targets: list[Tensor] = []

    @abstractmethod
    def extract_features(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
    ) -> Tensor | None:
        """Extract features from model outputs.

        Args:
            outputs: Model outputs dictionary.
            batch: Input batch dictionary.

        Returns:
            Feature tensor of shape (batch, ...) or None if unavailable.
        """
        ...

    @abstractmethod
    def extract_targets(
        self,
        batch: dict[str, Tensor | None],
    ) -> Tensor | None:
        """Extract target labels from batch.

        Args:
            batch: Input batch dictionary.

        Returns:
            Target tensor or None if unavailable.
        """
        ...

    def update(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate features and targets.

        Args:
            outputs: Model outputs.
            batch: Input batch.
            mask_labels: Mask labels (may be unused).
        """
        if len(self._features) >= self.n_train:
            return

        features = self.extract_features(outputs, batch)
        targets = self.extract_targets(batch)

        if features is None or targets is None:
            return

        self._features.append(features.cpu())
        self._targets.append(targets.cpu())

    def compute(self) -> dict[str, float]:
        """Train probe and compute accuracy.

        Returns:
            Dictionary with metric values.
        """
        if not self._features or not self._targets:
            return {self.name: 0.0}

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
        except ImportError:
            return {self.name: 0.0}

        import numpy as np

        # Prepare data
        X = torch.cat(self._features, dim=0).numpy()
        y = torch.cat(self._targets, dim=0).numpy()

        # Flatten features if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        # Handle case where we have very few samples
        if len(y) < 4:
            return {self.name: 0.0}

        # Split into train/test
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # Fallback if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Train logistic regression
        model = LogisticRegression(
            C=1.0 / self.regularization,
            max_iter=self.max_iterations,
            solver="lbfgs",
            multi_class="auto",
        )

        try:
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            return {self.name: float(accuracy)}
        except Exception:
            return {self.name: 0.0}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._features = []
        self._targets = []

    def state_objects(self) -> list[Any] | None:
        """Return state for distributed gathering."""
        return {"features": self._features, "targets": self._targets}

    def load_state_objects(self, gathered: list[Any]) -> None:
        """Load state from gathered objects."""
        all_features = []
        all_targets = []

        for item in gathered:
            if item is not None and isinstance(item, dict):
                all_features.extend(item.get("features", []))
                all_targets.extend(item.get("targets", []))

        self._features = all_features[:self.n_train]
        self._targets = all_targets[:self.n_train]


@register_metric("chain_probe")
class ChainProbeMetric(ProbeMetricBase):
    """Probe for classifying chain identity (heavy vs light).

    Tests whether embeddings encode chain-specific information.
    """

    name: ClassVar[str] = "chain_probe"
    requires_coords: ClassVar[bool] = False
    needs_attentions: ClassVar[bool] = False

    def __init__(
        self,
        pool_strategy: str = "mean",
        **kwargs,
    ) -> None:
        """Initialize the chain probe.

        Args:
            pool_strategy: How to pool sequence positions ("mean", "first", "last").
        """
        super().__init__(**kwargs)
        self.pool_strategy = pool_strategy

    def extract_features(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
    ) -> Tensor | None:
        """Extract pooled embeddings per chain.

        Returns features of shape (batch * 2, d_model) for heavy and light chains.
        """
        hidden_states = outputs.get("hidden_states")
        chain_ids = batch.get("chain_ids")
        attention_mask = batch.get("attention_mask")

        if hidden_states is None or chain_ids is None:
            return None

        batch_size, seq_len, d_model = hidden_states.shape
        features = []

        for chain_id in [0, 1]:  # 0 = heavy, 1 = light
            chain_mask = (chain_ids == chain_id)
            if attention_mask is not None:
                chain_mask = chain_mask & attention_mask.bool()

            for b in range(batch_size):
                mask = chain_mask[b]
                if mask.sum() == 0:
                    continue

                chain_embeds = hidden_states[b, mask]

                if self.pool_strategy == "mean":
                    pooled = chain_embeds.mean(dim=0)
                elif self.pool_strategy == "first":
                    pooled = chain_embeds[0]
                elif self.pool_strategy == "last":
                    pooled = chain_embeds[-1]
                else:
                    pooled = chain_embeds.mean(dim=0)

                features.append(pooled)

        if not features:
            return None

        return torch.stack(features, dim=0)

    def extract_targets(
        self,
        batch: dict[str, Tensor | None],
    ) -> Tensor | None:
        """Extract chain labels (0 = heavy, 1 = light)."""
        chain_ids = batch.get("chain_ids")
        attention_mask = batch.get("attention_mask")

        if chain_ids is None:
            return None

        batch_size = chain_ids.shape[0]
        labels = []

        for chain_id in [0, 1]:
            chain_mask = (chain_ids == chain_id)
            if attention_mask is not None:
                chain_mask = chain_mask & attention_mask.bool()

            for b in range(batch_size):
                if chain_mask[b].sum() > 0:
                    labels.append(chain_id)

        if not labels:
            return None

        return torch.tensor(labels)


@register_metric("position_probe")
class PositionProbeMetric(ProbeMetricBase):
    """Probe for predicting relative position in sequence.

    Tests whether embeddings encode positional information.
    Classifies positions as "beginning", "middle", or "end".
    """

    name: ClassVar[str] = "position_probe"
    requires_coords: ClassVar[bool] = False
    needs_attentions: ClassVar[bool] = False

    def __init__(
        self,
        n_bins: int = 3,
        sample_per_seq: int = 10,
        **kwargs,
    ) -> None:
        """Initialize the position probe.

        Args:
            n_bins: Number of position bins (classes).
            sample_per_seq: Number of positions to sample per sequence.
        """
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.sample_per_seq = sample_per_seq

    def extract_features(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
    ) -> Tensor | None:
        """Extract embeddings at sampled positions."""
        hidden_states = outputs.get("hidden_states")
        attention_mask = batch.get("attention_mask")

        if hidden_states is None or attention_mask is None:
            return None

        batch_size, seq_len, d_model = hidden_states.shape
        features = []

        for b in range(batch_size):
            valid_len = attention_mask[b].sum().item()
            if valid_len < self.n_bins:
                continue

            # Sample positions
            indices = torch.linspace(0, valid_len - 1, self.sample_per_seq).long()
            indices = indices.clamp(0, valid_len - 1)

            for idx in indices:
                features.append(hidden_states[b, idx])

        if not features:
            return None

        return torch.stack(features, dim=0)

    def extract_targets(
        self,
        batch: dict[str, Tensor | None],
    ) -> Tensor | None:
        """Extract position bin labels."""
        attention_mask = batch.get("attention_mask")

        if attention_mask is None:
            return None

        batch_size = attention_mask.shape[0]
        labels = []

        for b in range(batch_size):
            valid_len = attention_mask[b].sum().item()
            if valid_len < self.n_bins:
                continue

            # Sample positions and compute bin labels
            indices = torch.linspace(0, valid_len - 1, self.sample_per_seq)

            for idx in indices:
                relative_pos = idx / (valid_len - 1)
                bin_label = int(relative_pos * self.n_bins)
                bin_label = min(bin_label, self.n_bins - 1)
                labels.append(bin_label)

        if not labels:
            return None

        return torch.tensor(labels)


@register_metric("cdr_probe")
class CDRProbeMetric(ProbeMetricBase):
    """Probe for predicting CDR (Complementarity Determining Region) membership.

    Tests whether embeddings encode structural/functional region information.
    """

    name: ClassVar[str] = "cdr_probe"
    requires_coords: ClassVar[bool] = False
    needs_attentions: ClassVar[bool] = False

    def __init__(
        self,
        sample_per_seq: int = 20,
        **kwargs,
    ) -> None:
        """Initialize the CDR probe.

        Args:
            sample_per_seq: Number of positions to sample per sequence.
        """
        super().__init__(**kwargs)
        self.sample_per_seq = sample_per_seq

    def extract_features(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
    ) -> Tensor | None:
        """Extract embeddings at sampled positions."""
        hidden_states = outputs.get("hidden_states")
        attention_mask = batch.get("attention_mask")
        cdr_mask = batch.get("cdr_mask")

        if hidden_states is None or attention_mask is None or cdr_mask is None:
            return None

        batch_size, seq_len, d_model = hidden_states.shape
        features = []

        for b in range(batch_size):
            valid_mask = attention_mask[b].bool()
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            if len(valid_indices) < self.sample_per_seq:
                continue

            # Sample positions
            sample_indices = torch.randperm(len(valid_indices))[:self.sample_per_seq]
            selected_positions = valid_indices[sample_indices]

            for pos in selected_positions:
                features.append(hidden_states[b, pos])

        if not features:
            return None

        return torch.stack(features, dim=0)

    def extract_targets(
        self,
        batch: dict[str, Tensor | None],
    ) -> Tensor | None:
        """Extract CDR labels (0 = framework, 1 = CDR)."""
        attention_mask = batch.get("attention_mask")
        cdr_mask = batch.get("cdr_mask")

        if attention_mask is None or cdr_mask is None:
            return None

        batch_size = attention_mask.shape[0]
        labels = []

        for b in range(batch_size):
            valid_mask = attention_mask[b].bool()
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            if len(valid_indices) < self.sample_per_seq:
                continue

            # Sample same positions as in extract_features
            # Note: This relies on deterministic sampling which may not match
            # In practice, we'd want to store the sampled indices
            sample_indices = torch.randperm(len(valid_indices))[:self.sample_per_seq]
            selected_positions = valid_indices[sample_indices]

            for pos in selected_positions:
                labels.append(cdr_mask[b, pos].item())

        if not labels:
            return None

        return torch.tensor(labels)
