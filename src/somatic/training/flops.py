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

    Formula breakdown per transformer layer:

    Attention:
    - Standard MHA: Q, K, V projections = 3 * 2 * d_model^2 per token
    - Chain-aware attention: doubles QKV = 6 * 2 * d_model^2 per token
    - QK^T matmul = 2 * seq_len * d_model per token
    - Attention @ V = 2 * seq_len * d_model per token
    - Output projection = 2 * d_model^2 per token

    FFN (SwiGLU):
    - Fused gate+up = 2 * d_model * (2 * d_ffn) per token
    - Down projection = 2 * d_ffn * d_model per token
    - Total = 6 * d_model * d_ffn per token

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

    # QKV projection factor: 1x for standard attention, 2x for chain-aware
    # Chain-aware attention has separate QKV projections for self and cross attention
    qkv_factor = 2 if config.use_chain_aware_attention else 1

    flops_per_layer = 0

    # Attention QKV projections: qkv_factor * 3 * 2 * d_model^2 per token
    # Each linear layer: 2 * input * output FLOPs (multiply-add pairs)
    # 3 projections (Q, K, V) each with d_model -> d_model
    flops_per_layer += qkv_factor * 3 * 2 * d_model * d_model

    # Attention QK^T and Attn@V: 2 * 2 * seq_len * d_model per token
    # QK^T: 2 * seq_len * d_model (query @ key.T per position)
    # Attn@V: 2 * seq_len * d_model (attention weights @ values per position)
    flops_per_layer += 4 * seq_len * d_model

    # Output projection: 2 * d_model^2 per token
    # Linear layer: d_model -> d_model
    flops_per_layer += 2 * d_model * d_model

    # FFN (SwiGLU): 3 * 2 * d_model * d_ffn per token
    # SwiGLU has 3 weight matrices: gate_up (d_model -> 2*d_ffn) and down (d_ffn -> d_model)
    # Equivalent to: 2 * d_model * 2*d_ffn + 2 * d_ffn * d_model = 6 * d_model * d_ffn
    flops_per_layer += 6 * d_model * d_ffn

    # Total for all layers
    total_flops = n_layers * flops_per_layer

    # LM head: 2 * d_model * vocab_size per token
    # Linear layer: d_model -> vocab_size
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
