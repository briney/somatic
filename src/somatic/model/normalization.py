"""Normalization layers and utilities for Somatic transformer."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm simplifies LayerNorm by removing the mean-centering operation,
    providing computational savings while maintaining effectiveness.

    Reference: https://arxiv.org/abs/1910.07467

    Args:
        normalized_shape: Size of the last dimension to normalize
        eps: Epsilon for numerical stability
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight


class LearnedQKScale(nn.Module):
    """
    Per-head learned scaling for Q and K in attention.

    Applies learnable per-head scaling factors to queries and keys
    before computing attention scores.

    Args:
        n_heads: Number of attention heads
        head_dim: Dimension per head
    """

    def __init__(self, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        # Initialize to 1.0 (identity)
        self.q_scale = nn.Parameter(torch.ones(n_heads, 1, 1))
        self.k_scale = nn.Parameter(torch.ones(n_heads, 1, 1))

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply learned scaling to Q and K.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, n_heads, seq_len, head_dim)

        Returns:
            Scaled (q, k) tensors
        """
        return q * self.q_scale, k * self.k_scale


class QKNormModule(nn.Module):
    """
    Applies normalization to Q and K tensors.

    Args:
        norm_type: Type of normalization ("layernorm" or "rmsnorm")
        head_dim: Dimension per head
        eps: Epsilon for numerical stability
    """

    def __init__(
        self,
        norm_type: str,
        head_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.q_norm = create_norm_layer(norm_type, head_dim, eps)
        self.k_norm = create_norm_layer(norm_type, head_dim, eps)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply normalization to Q and K.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, n_heads, seq_len, head_dim)

        Returns:
            Normalized (q, k) tensors
        """
        return self.q_norm(q), self.k_norm(k)


def create_norm_layer(
    norm_type: str,
    normalized_shape: int,
    eps: float = 1e-6,
) -> nn.Module:
    """
    Factory function to create normalization layers.

    Args:
        norm_type: Type of normalization ("layernorm" or "rmsnorm")
        normalized_shape: Size of the last dimension to normalize
        eps: Epsilon for numerical stability

    Returns:
        Normalization layer instance

    Raises:
        ValueError: If norm_type is not recognized
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(normalized_shape, eps=eps)
    elif norm_type == "rmsnorm":
        return RMSNorm(normalized_shape, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


def create_qk_norm(
    qk_norm_type: str,
    norm_type: str,
    n_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> nn.Module | None:
    """
    Factory function to create QK normalization.

    Args:
        qk_norm_type: Type of QK normalization ("none", "norm", or "learned_scale")
        norm_type: Base norm type for "norm" mode ("layernorm" or "rmsnorm")
        n_heads: Number of attention heads
        head_dim: Dimension per head
        eps: Epsilon for numerical stability

    Returns:
        QK normalization module or None if qk_norm_type is "none"

    Raises:
        ValueError: If qk_norm_type is not recognized
    """
    if qk_norm_type == "none":
        return None
    elif qk_norm_type == "norm":
        return QKNormModule(norm_type, head_dim, eps)
    elif qk_norm_type == "learned_scale":
        return LearnedQKScale(n_heads, head_dim)
    else:
        raise ValueError(f"Unknown qk_norm_type: {qk_norm_type}")
