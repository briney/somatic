"""FLOPs tracking for training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import SomaticConfig


@dataclass
class FLOPsConfig:
    """Configuration for FLOPs tracking.

    Parameters
    ----------
    enabled
        Master switch to enable/disable FLOPs tracking.
    """

    enabled: bool = True


def compute_model_flops_per_token(config: SomaticConfig) -> int:
    """Compute FLOPs per token for forward pass.

    Per-layer FLOPs/token are decomposed into three attention factors that
    differ across modes:

    - ``projection_factor``: number of Q/K/V projection matmuls per layer.
    - ``score_factor``: number of dense QK^T score matmuls per layer.
    - ``value_factor``: number of (attn @ V) value matmuls per layer.

    Mode-specific factors:

    - Standard MHA: projection=3, score=1, value=1.
    - Separate-QKV chain-aware attention: projection=6, score=2, value=2.
    - Shared-QKV chain-aware attention: projection=3, score=2, value=1.

    Per-layer FLOPs/token:

    - Projections: ``projection_factor * 2 * d_model^2``
    - QK^T: ``score_factor * 2 * seq_len * d_model``
    - Attn @ V: ``value_factor * 2 * seq_len * d_model``
    - Output projection: ``2 * d_model^2``
    - SwiGLU FFN: ``6 * d_model * d_ffn``

    Parameters
    ----------
    config
        Model configuration with architecture parameters.

    Returns
    -------
    int
        Estimated FLOPs per token for a single forward pass.
    """
    d_model = config.d_model
    n_layers = config.n_layers
    d_ffn = config.d_ffn
    seq_len = config.max_seq_len
    vocab_size = config.vocab_size

    if not config.use_chain_aware_attention:
        projection_factor, score_factor, value_factor = 3, 1, 1
    elif config.chain_aware_projection_mode == "shared":
        projection_factor, score_factor, value_factor = 3, 2, 1
    else:  # "separate"
        projection_factor, score_factor, value_factor = 6, 2, 2

    flops_per_layer = 0

    # Q/K/V projections (each linear: 2 * input * output FLOPs per token).
    flops_per_layer += projection_factor * 2 * d_model * d_model

    # QK^T scores (per token, dense over seq_len keys).
    flops_per_layer += score_factor * 2 * seq_len * d_model

    # Attn @ V (per token, dense over seq_len values).
    flops_per_layer += value_factor * 2 * seq_len * d_model

    # Output projection: d_model -> d_model.
    flops_per_layer += 2 * d_model * d_model

    # FFN (SwiGLU): gate_up (d_model -> 2*d_ffn) and down (d_ffn -> d_model)
    # collapse to 6 * d_model * d_ffn FLOPs per token.
    flops_per_layer += 6 * d_model * d_ffn

    total_flops = n_layers * flops_per_layer

    # LM head: d_model -> vocab_size.
    total_flops += 2 * d_model * vocab_size

    return total_flops


def compute_training_flops_per_token(config: SomaticConfig) -> int:
    """Compute FLOPs per token for forward + backward pass.

    Standard approximation: backward pass ~= 2x forward pass FLOPs,
    so total training step = 3x forward pass.

    Parameters
    ----------
    config
        Model configuration.

    Returns
    -------
    int
        FLOPs per token for training (forward + backward).
    """
    return 3 * compute_model_flops_per_token(config)


class FLOPsTracker:
    """Tracks cumulative FLOPs during training.

    Parameters
    ----------
    config
        FLOPs tracking configuration.
    model_config
        Model architecture configuration for FLOPs calculation.
    world_size
        Number of GPUs for distributed training.
    """

    def __init__(
        self,
        config: FLOPsConfig,
        model_config: SomaticConfig,
        world_size: int = 1,
    ) -> None:
        self.config = config
        self._flops_per_token = compute_training_flops_per_token(model_config)
        self.world_size = world_size
        self._cumulative_flops: int = 0

    def update(self, batch_size: int, seq_len: int) -> None:
        """Update cumulative FLOPs after a training step.

        Parameters
        ----------
        batch_size
            Number of sequences in the batch.
        seq_len
            Sequence length (tokens per sequence).
        """
        if not self.config.enabled:
            return

        # Tokens processed this step (across all GPUs)
        tokens = batch_size * seq_len * self.world_size

        # Accumulate FLOPs
        self._cumulative_flops += tokens * self._flops_per_token

    def compute(self) -> dict[str, float]:
        """Compute metrics for logging.

        Returns
        -------
        dict[str, float]
            Dictionary with cumulative_flops.
        """
        if not self.config.enabled:
            return {}

        return {"cumulative_flops": float(self._cumulative_flops)}

    @property
    def flops_per_token(self) -> int:
        """FLOPs per token for training."""
        return self._flops_per_token
