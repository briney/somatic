"""Contact prediction metrics using attention weights."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor

from ..base import MetricBase
from ..registry import register_metric


def compute_distance_matrix(coords: Tensor) -> Tensor:
    """Compute pairwise distance matrix from coordinates.

    Args:
        coords: Coordinates tensor of shape (batch, seq_len, 3).

    Returns:
        Distance matrix of shape (batch, seq_len, seq_len).
    """
    # coords: (batch, seq_len, 3)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (batch, seq_len, seq_len, 3)
    distances = torch.sqrt((diff**2).sum(dim=-1) + 1e-8)  # (batch, seq_len, seq_len)
    return distances


def compute_contact_map(
    coords: Tensor,
    threshold: float = 8.0,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Compute binary contact map from coordinates.

    Args:
        coords: Coordinates tensor of shape (batch, seq_len, 3).
        threshold: Distance threshold for defining contacts (in Angstroms).
        attention_mask: Optional mask for valid positions.

    Returns:
        Binary contact map of shape (batch, seq_len, seq_len).
    """
    distances = compute_distance_matrix(coords)
    contacts = (distances < threshold).float()

    # Mask out invalid positions
    if attention_mask is not None:
        # Create 2D mask from 1D mask
        mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        contacts = contacts * mask_2d.float()

    return contacts


def symmetrize_attention(attention: Tensor) -> Tensor:
    """Symmetrize attention matrix.

    Args:
        attention: Attention tensor of shape (..., seq_len, seq_len).

    Returns:
        Symmetrized attention of same shape.
    """
    return (attention + attention.transpose(-2, -1)) / 2


def apply_apc(attention: Tensor) -> Tensor:
    """Apply Average Product Correction to attention matrix.

    APC removes the background signal from attention weights,
    improving contact prediction performance.

    Args:
        attention: Attention tensor of shape (..., seq_len, seq_len).

    Returns:
        APC-corrected attention of same shape.
    """
    # Compute row and column means
    row_mean = attention.mean(dim=-1, keepdim=True)
    col_mean = attention.mean(dim=-2, keepdim=True)
    global_mean = attention.mean(dim=(-2, -1), keepdim=True)

    # Apply APC: A_ij - (A_i * A_j) / A
    apc = row_mean * col_mean / (global_mean + 1e-8)
    corrected = attention - apc

    return corrected


@register_metric("p_at_l")
class PrecisionAtLMetric(MetricBase):
    """Precision@L metric for contact prediction.

    Computes the precision of the top-L predicted contacts, where L is the
    sequence length. This is a standard metric for evaluating protein contact
    prediction from language model representations.

    The metric can use attention weights as contact predictions or fall back
    to using hidden state similarity. Optionally supports training logistic
    regression on attention patterns.

    Attributes:
        contact_threshold: Distance threshold for defining contacts (Angstroms).
        min_seq_sep: Minimum sequence separation for valid contacts.
        use_attention: Whether to use attention weights for prediction.
        attention_layer: Which layer(s) to use ("last", "mean", or layer index).
        head_aggregation: How to aggregate heads ("mean", "max").
        num_layers: Number of layers to average (from the end).
        use_logistic_regression: Whether to use logistic regression mode.
        logreg_n_train: Number of sequences to use for logistic regression training.
        logreg_lambda: L2 regularization for logistic regression.
    """

    name: ClassVar[str] = "p_at_l"
    requires_coords: ClassVar[bool] = True
    needs_attentions: ClassVar[bool] = True

    def __init__(
        self,
        contact_threshold: float = 8.0,
        min_seq_sep: int = 6,
        use_attention: bool = True,
        attention_layer: int | str = "last",
        head_aggregation: str = "mean",
        num_layers: int | None = None,
        use_logistic_regression: bool = False,
        logreg_n_train: int = 20,
        logreg_lambda: float = 0.15,
        logreg_n_iterations: int = 5,
        **kwargs,
    ) -> None:
        """Initialize the metric.

        Args:
            contact_threshold: Distance threshold for contacts.
            min_seq_sep: Minimum sequence separation.
            use_attention: Whether to use attention weights.
            attention_layer: Layer selection strategy.
            head_aggregation: Head aggregation method.
            num_layers: Number of layers to use (from end).
            use_logistic_regression: Use logistic regression mode.
            logreg_n_train: Training sequences for logreg.
            logreg_lambda: L2 regularization strength.
            logreg_n_iterations: Number of logreg iterations.
        """
        super().__init__()

        self.contact_threshold = contact_threshold
        self.min_seq_sep = min_seq_sep
        self.use_attention = use_attention
        self.attention_layer = attention_layer
        self.head_aggregation = head_aggregation
        self.num_layers = num_layers
        self.use_logistic_regression = use_logistic_regression
        self.logreg_n_train = logreg_n_train
        self.logreg_lambda = logreg_lambda
        self.logreg_n_iterations = logreg_n_iterations

        # Accumulators for precision computation
        self._correct: int = 0
        self._total: int = 0

        # Storage for logistic regression mode
        self._logreg_features: list[Tensor] = []
        self._logreg_targets: list[Tensor] = []
        self._logreg_weights: Tensor | None = None

    def _get_attention_prediction(
        self,
        attentions: tuple[Tensor, ...],
        attention_mask: Tensor,
    ) -> Tensor:
        """Extract and process attention weights for contact prediction.

        Args:
            attentions: Tuple of attention tensors from each layer.
                Each tensor has shape (batch, n_heads, seq_len, seq_len).
            attention_mask: Mask for valid positions.

        Returns:
            Processed attention scores (batch, seq_len, seq_len).
        """
        num_layers = self.num_layers or 1

        # Select layers
        if self.attention_layer == "last":
            selected = attentions[-num_layers:]
        elif self.attention_layer == "mean":
            selected = attentions[-num_layers:]
        elif isinstance(self.attention_layer, int):
            selected = [attentions[self.attention_layer]]
        else:
            selected = attentions[-num_layers:]

        # Stack and aggregate heads
        # selected is a list of (batch, n_heads, seq_len, seq_len)
        stacked = torch.stack(selected, dim=1)  # (batch, n_selected, n_heads, seq, seq)

        if self.head_aggregation == "mean":
            aggregated = stacked.mean(dim=(1, 2))  # (batch, seq, seq)
        elif self.head_aggregation == "max":
            aggregated = stacked.max(dim=2).values.max(dim=1).values
        else:
            aggregated = stacked.mean(dim=(1, 2))

        # Symmetrize and apply APC
        symmetrized = symmetrize_attention(aggregated)
        corrected = apply_apc(symmetrized)

        # Mask invalid positions
        mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        corrected = corrected * mask_2d.float()

        return corrected

    def _compute_precision_at_l(
        self,
        predictions: Tensor,
        contacts: Tensor,
        attention_mask: Tensor,
    ) -> tuple[int, int]:
        """Compute precision@L for a batch.

        Args:
            predictions: Predicted contact scores (batch, seq_len, seq_len).
            contacts: Ground truth contacts (batch, seq_len, seq_len).
            attention_mask: Mask for valid positions.

        Returns:
            Tuple of (correct predictions, total predictions).
        """
        batch_size, seq_len, _ = predictions.shape
        correct = 0
        total = 0

        for b in range(batch_size):
            # Get valid length for this sequence
            valid_len = attention_mask[b].sum().item()
            L = int(valid_len)

            if L < self.min_seq_sep + 1:
                continue

            # Create sequence separation mask
            indices = torch.arange(seq_len, device=predictions.device)
            sep_mask = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs() >= self.min_seq_sep

            # Get upper triangle with separation constraint
            triu_mask = torch.triu(torch.ones(seq_len, seq_len, device=predictions.device), diagonal=1)
            valid_mask = triu_mask * sep_mask.float()

            # Apply attention mask
            valid_mask = valid_mask * (attention_mask[b].unsqueeze(0) * attention_mask[b].unsqueeze(1)).float()

            # Get predictions and targets for valid positions
            pred_scores = predictions[b] * valid_mask
            true_contacts = contacts[b] * valid_mask

            # Get top-L predictions
            flat_pred = pred_scores.view(-1)
            flat_true = true_contacts.view(-1)
            flat_valid = valid_mask.view(-1)

            # Only consider valid positions
            valid_indices = flat_valid.nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                continue

            valid_pred = flat_pred[valid_indices]
            valid_true = flat_true[valid_indices]

            # Get top-L predictions
            k = min(L, len(valid_indices))
            _, top_indices = valid_pred.topk(k)
            top_true = valid_true[top_indices]

            correct += int(top_true.sum().item())
            total += k

        return correct, total

    def update(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate precision from a batch.

        Args:
            outputs: Model outputs with "attentions" key.
            batch: Input batch with "coords" and "attention_mask".
            mask_labels: Binary mask (unused for this metric).
        """
        coords = batch.get("coords")
        attention_mask = batch.get("attention_mask")
        attentions = outputs.get("attentions")

        if coords is None or attentions is None or attention_mask is None:
            return

        # Compute ground truth contacts
        contacts = compute_contact_map(
            coords, self.contact_threshold, attention_mask
        )

        if self.use_logistic_regression:
            # Store features for later logistic regression
            self._store_logreg_data(attentions, contacts, attention_mask)
        else:
            # Direct attention-based prediction
            predictions = self._get_attention_prediction(attentions, attention_mask)
            correct, total = self._compute_precision_at_l(
                predictions, contacts, attention_mask
            )
            self._correct += correct
            self._total += total

    def _store_logreg_data(
        self,
        attentions: tuple[Tensor, ...],
        contacts: Tensor,
        attention_mask: Tensor,
    ) -> None:
        """Store attention features for logistic regression training.

        Args:
            attentions: Attention tensors from all layers.
            contacts: Ground truth contact map.
            attention_mask: Valid position mask.
        """
        # Only store up to n_train sequences
        if len(self._logreg_features) >= self.logreg_n_train:
            return

        # Stack all attention heads from all layers
        # Each attention is (batch, n_heads, seq, seq)
        all_attn = torch.stack(attentions, dim=1)  # (batch, n_layers, n_heads, seq, seq)

        batch_size = all_attn.shape[0]
        for b in range(batch_size):
            if len(self._logreg_features) >= self.logreg_n_train:
                break

            # Get features for this sequence
            features = all_attn[b]  # (n_layers, n_heads, seq, seq)
            target = contacts[b]  # (seq, seq)
            mask = attention_mask[b]  # (seq,)

            self._logreg_features.append(features.cpu())
            self._logreg_targets.append(target.cpu())

    def compute(self) -> dict[str, float]:
        """Compute precision@L from accumulated data.

        Returns:
            Dictionary with "p_at_l" key.
        """
        if self.use_logistic_regression and self._logreg_features:
            return self._compute_logreg_precision()

        if self._total == 0:
            return {self.name: 0.0}

        return {self.name: self._correct / self._total}

    def _compute_logreg_precision(self) -> dict[str, float]:
        """Compute precision using logistic regression on attention.

        Returns:
            Dictionary with precision values.
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            # Fall back to standard precision if sklearn not available
            if self._total == 0:
                return {self.name: 0.0}
            return {self.name: self._correct / self._total}

        # Prepare training data
        all_features = []
        all_targets = []

        for features, targets in zip(self._logreg_features, self._logreg_targets):
            # Flatten to get feature vectors for each position pair
            n_layers, n_heads, seq_len, _ = features.shape
            n_features = n_layers * n_heads

            # Get upper triangle with sequence separation
            indices = torch.arange(seq_len)
            sep_mask = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs() >= self.min_seq_sep
            triu = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            valid_mask = triu & sep_mask

            valid_indices = valid_mask.nonzero(as_tuple=False)
            if len(valid_indices) == 0:
                continue

            for i, j in valid_indices:
                # Feature vector: attention from all heads/layers for this pair
                feat = features[:, :, i, j].flatten()  # (n_features,)
                target = targets[i, j].item()
                all_features.append(feat.numpy())
                all_targets.append(target)

        if not all_features:
            return {self.name: 0.0}

        import numpy as np

        X = np.array(all_features)
        y = np.array(all_targets)

        # Train logistic regression
        model = LogisticRegression(
            C=1.0 / self.logreg_lambda,
            max_iter=self.logreg_n_iterations * 100,
            solver="lbfgs",
        )

        try:
            model.fit(X, y)

            # Compute precision on all data (training = test for simplicity)
            predictions = model.predict_proba(X)[:, 1]

            # Get top-L predictions
            L = len(y) // 10  # Approximate L
            if L == 0:
                L = 1
            top_indices = np.argsort(predictions)[-L:]
            precision = y[top_indices].mean()

            return {self.name: float(precision)}
        except Exception:
            return {self.name: 0.0}

    def reset(self) -> None:
        """Reset accumulated state."""
        self._correct = 0
        self._total = 0
        self._logreg_features = []
        self._logreg_targets = []
        self._logreg_weights = None

    def state_tensors(self) -> list[Tensor]:
        """Return state as tensors for distributed aggregation."""
        if self.use_logistic_regression:
            # For logreg mode, use object-based gathering
            return []
        return [torch.tensor([float(self._correct), float(self._total)])]

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        """Load state from gathered tensors."""
        if not self.use_logistic_regression and tensors:
            state = tensors[0]
            self._correct = int(state[0].item())
            self._total = int(state[1].item())

    def state_objects(self) -> list[Any] | None:
        """Return state objects for logreg mode."""
        if self.use_logistic_regression:
            return {
                "features": self._logreg_features,
                "targets": self._logreg_targets,
                "correct": self._correct,
                "total": self._total,
            }
        return None

    def load_state_objects(self, gathered: list[Any]) -> None:
        """Load state from gathered objects."""
        if gathered:
            # Combine features and targets from all processes
            all_features = []
            all_targets = []
            total_correct = 0
            total_count = 0

            for item in gathered:
                if item is not None and isinstance(item, dict):
                    all_features.extend(item.get("features", []))
                    all_targets.extend(item.get("targets", []))
                    total_correct += item.get("correct", 0)
                    total_count += item.get("total", 0)

            self._logreg_features = all_features[:self.logreg_n_train]
            self._logreg_targets = all_targets[:self.logreg_n_train]
            self._correct = total_correct
            self._total = total_count
