"""
Attention modules for Somatic transformer.

This module provides three attention implementations:
1. MultiHeadAttention: Standard self-attention with RoPE and SDPA optimization
2. ChainAwareAttention: MINT-style hybrid intra/inter-chain attention with
   separate Q/K/V projections for self and cross paths
3. SharedQKVChainAwareAttention: Chain-aware attention that shares one Q/K/V
   projection but routes scores through RoPE (intra-chain) vs no-position
   (inter-chain) before a single global softmax
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .normalization import create_qk_norm, create_qkv_norm
from .rope import RotaryPositionEmbedding


class BaseAttention(nn.Module):
    """Base class for attention modules with shared initialization and utilities."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 512,
        qk_norm: str = "none",
        norm_type: str = "layernorm",
        layer_norm_eps: float = 1e-6,
        hybrid_norm: bool = False,
        rope_fraction: float = 1.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.dropout_p = dropout
        self.inner_dim = n_heads * head_dim
        self.hybrid_norm = hybrid_norm

        # Output projection (shared by all attention types)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=bias)

        # RoPE (with optional partial rotation; fraction=1.0 = full RoPE)
        self.rope = RotaryPositionEmbedding(
            head_dim, max_seq_len=max_seq_len, fraction=rope_fraction
        )

        self.dropout = nn.Dropout(dropout)

        # QK normalization is disabled in hybrid_norm mode (subclasses build QKV norm instead)
        if hybrid_norm:
            self.qk_norm_module = None
        else:
            self.qk_norm_module = create_qk_norm(
                qk_norm, norm_type, n_heads, head_dim, layer_norm_eps
            )

    def _apply_qk_norm(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply QK normalization if configured."""
        if self.qk_norm_module is not None:
            return self.qk_norm_module(q, k)
        return q, k

    def _create_padding_mask(
        self, attention_mask: Tensor | None, input_dtype: torch.dtype
    ) -> Tensor | None:
        """
        Create additive padding mask for attention.

        Args:
            attention_mask: Optional padding mask of shape (batch, seq_len)
            input_dtype: Dtype of input tensor (for mixed precision compatibility)

        Returns:
            Additive mask with -inf for padding (batch, 1, 1, seq_len) or None
        """
        if attention_mask is None:
            return None

        # Create additive mask: 0 where valid, -inf where padding
        # Use same dtype as input for mixed precision compatibility
        padding_mask = torch.zeros_like(attention_mask, dtype=input_dtype)
        padding_mask = padding_mask.masked_fill(~attention_mask.bool(), float("-inf"))
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        return padding_mask


class MultiHeadAttention(BaseAttention):
    """
    Standard multi-head self-attention with RoPE.

    Uses F.scaled_dot_product_attention for efficiency when need_weights=False,
    allowing PyTorch to use optimized implementations (Flash Attention, etc.).

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        head_dim: Dimension per head (default: 64)
        dropout: Attention dropout probability
        bias: Whether to use bias in projections
        max_seq_len: Maximum sequence length for RoPE
        qk_norm: QK normalization type ("none", "norm", or "learned_scale")
        norm_type: Normalization type for qk_norm="norm" ("layernorm" or "rmsnorm")
        layer_norm_eps: Epsilon for normalization layers
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 512,
        qk_norm: str = "none",
        norm_type: str = "layernorm",
        layer_norm_eps: float = 1e-6,
        hybrid_norm: bool = False,
        rope_fraction: float = 1.0,
    ) -> None:
        super().__init__(
            d_model, n_heads, head_dim, dropout, bias, max_seq_len,
            qk_norm, norm_type, layer_norm_eps, hybrid_norm,
            rope_fraction=rope_fraction,
        )

        # QKV projections
        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=bias)

        # HybridNorm: QKV-norm in place of QK-norm
        self.qkv_norm: nn.Module | None = None
        if hybrid_norm:
            self.qkv_norm = create_qkv_norm(norm_type, head_dim, layer_norm_eps)

    def forward(
        self,
        x: Tensor,
        chain_ids: Tensor,
        attention_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass with standard self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            chain_ids: Chain identity tensor of shape (batch, seq_len) - ignored
            attention_mask: Optional padding mask of shape (batch, seq_len)
            need_weights: If True, return attention weights (disables SDPA)

        Returns:
            If need_weights is False:
                Output tensor of shape (batch, seq_len, d_model)
            If need_weights is True:
                Tuple of (output, attn_weights) where attn_weights has shape
                (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=self.n_heads)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Apply QKV/QK normalization
        if self.qkv_norm is not None:
            q, k, v = self.qkv_norm(q, k, v)
        else:
            q, k = self._apply_qk_norm(q, k)

        if need_weights:
            # Manual attention computation to get weights
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply padding mask
            padding_mask = self._create_padding_mask(attention_mask, x.dtype)
            if padding_mask is not None:
                scores = scores + padding_mask

            # Softmax and dropout
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_weights = self.dropout(attn_weights)

            # Compute output
            output = torch.matmul(attn_weights, v)
        else:
            # Use efficient SDPA
            padding_mask = self._create_padding_mask(attention_mask, x.dtype)

            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=padding_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=self.scale,
            )
            attn_weights = None

        # Reshape and project
        output = rearrange(output, "b h s d -> b s (h d)")
        output = self.out_proj(output)

        if need_weights:
            return output, attn_weights
        return output


def _create_chain_mask(chain_ids: Tensor) -> Tensor:
    """Create intra-chain boolean mask of shape (batch, 1, seq_len, seq_len)."""
    chain_i = chain_ids.unsqueeze(-1)
    chain_j = chain_ids.unsqueeze(-2)
    return (chain_i == chain_j).unsqueeze(1)


class ChainAwareAttention(BaseAttention):
    """
    Attention module implementing MINT-style hybrid intra/inter-chain attention.

    For antibody sequences with multiple chains:
    1. Computes self-attention scores (with RoPE) for intra-chain pairs
    2. Computes cross-attention scores (without RoPE) for inter-chain pairs
    3. Merges scores before softmax: intra-chain uses self scores, inter-chain uses cross scores
    4. Applies single softmax to merged scores
    5. Splits attention weights after softmax for value multiplication

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        head_dim: Dimension per head (default: 64)
        dropout: Attention dropout probability
        bias: Whether to use bias in projections
        max_seq_len: Maximum sequence length for RoPE
        qk_norm: QK normalization type ("none", "norm", or "learned_scale")
        norm_type: Normalization type for qk_norm="norm" ("layernorm" or "rmsnorm")
        layer_norm_eps: Epsilon for normalization layers
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 512,
        qk_norm: str = "none",
        norm_type: str = "layernorm",
        layer_norm_eps: float = 1e-6,
        hybrid_norm: bool = False,
        rope_fraction: float = 1.0,
    ) -> None:
        # Don't pass qk_norm to base class - we handle it separately for self/cross paths
        super().__init__(
            d_model, n_heads, head_dim, dropout, bias, max_seq_len,
            qk_norm="none", norm_type=norm_type, layer_norm_eps=layer_norm_eps,
            hybrid_norm=hybrid_norm,
            rope_fraction=rope_fraction,
        )

        # Self-attention projections (RoPE will be applied to Q and K)
        self.q_self = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.k_self = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.v_self = nn.Linear(d_model, self.inner_dim, bias=bias)

        # Cross-attention projections (no RoPE)
        self.q_cross = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.k_cross = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.v_cross = nn.Linear(d_model, self.inner_dim, bias=bias)

        # Normalization: HybridNorm uses QKV-norm; otherwise QK-norm (per path)
        self.qkv_norm_self: nn.Module | None = None
        self.qkv_norm_cross: nn.Module | None = None
        self.qk_norm_self: nn.Module | None = None
        self.qk_norm_cross: nn.Module | None = None
        if hybrid_norm:
            self.qkv_norm_self = create_qkv_norm(norm_type, head_dim, layer_norm_eps)
            self.qkv_norm_cross = create_qkv_norm(norm_type, head_dim, layer_norm_eps)
        else:
            self.qk_norm_self = create_qk_norm(
                qk_norm, norm_type, n_heads, head_dim, layer_norm_eps
            )
            self.qk_norm_cross = create_qk_norm(
                qk_norm, norm_type, n_heads, head_dim, layer_norm_eps
            )

    def _create_chain_mask(self, chain_ids: Tensor) -> Tensor:
        """
        Create intra-chain boolean mask.

        Args:
            chain_ids: Chain identity tensor of shape (batch, seq_len)

        Returns:
            intra_mask: Boolean mask where True = same chain (batch, 1, seq_len, seq_len)
        """
        return _create_chain_mask(chain_ids)

    def forward(
        self,
        x: Tensor,
        chain_ids: Tensor,
        attention_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass with MINT-style chain-aware attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            chain_ids: Chain identity tensor of shape (batch, seq_len)
            attention_mask: Optional padding mask of shape (batch, seq_len)
            need_weights: If True, return attention weights

        Returns:
            If need_weights is False:
                Output tensor of shape (batch, seq_len, d_model)
            If need_weights is True:
                Tuple of (output, attn_weights) where attn_weights has shape
                (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V for both attention types
        q_self = rearrange(self.q_self(x), "b s (h d) -> b h s d", h=self.n_heads)
        k_self = rearrange(self.k_self(x), "b s (h d) -> b h s d", h=self.n_heads)
        v_self = rearrange(self.v_self(x), "b s (h d) -> b h s d", h=self.n_heads)

        q_cross = rearrange(self.q_cross(x), "b s (h d) -> b h s d", h=self.n_heads)
        k_cross = rearrange(self.k_cross(x), "b s (h d) -> b h s d", h=self.n_heads)
        v_cross = rearrange(self.v_cross(x), "b s (h d) -> b h s d", h=self.n_heads)

        # Apply RoPE only to self-attention Q and K (not cross-attention)
        q_self, k_self = self.rope(q_self, k_self)

        # Apply normalization (QKV in HybridNorm mode, otherwise QK)
        if self.qkv_norm_self is not None:
            q_self, k_self, v_self = self.qkv_norm_self(q_self, k_self, v_self)
        elif self.qk_norm_self is not None:
            q_self, k_self = self.qk_norm_self(q_self, k_self)
        if self.qkv_norm_cross is not None:
            q_cross, k_cross, v_cross = self.qkv_norm_cross(q_cross, k_cross, v_cross)
        elif self.qk_norm_cross is not None:
            q_cross, k_cross = self.qk_norm_cross(q_cross, k_cross)

        # Compute raw attention scores
        scores_self = torch.matmul(q_self, k_self.transpose(-2, -1)) * self.scale
        scores_cross = torch.matmul(q_cross, k_cross.transpose(-2, -1)) * self.scale

        # Create masks
        intra_mask = self._create_chain_mask(chain_ids)
        padding_mask = self._create_padding_mask(attention_mask, x.dtype)

        # Convert intra_mask to same dtype as input for torch.where and later multiplication
        intra_mask_float = intra_mask.to(x.dtype)

        # Merge attention scores before softmax:
        # Use self scores for intra-chain pairs, cross scores for inter-chain pairs
        merged_scores = torch.where(intra_mask, scores_self, scores_cross)

        # Apply padding mask
        if padding_mask is not None:
            merged_scores = merged_scores + padding_mask

        # Single softmax over all positions
        attn_weights = F.softmax(merged_scores, dim=-1)

        # Handle NaN from all-masked rows (e.g., all padding)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Compute weighted values using chain mask to route to appropriate values
        # Intra-chain positions use v_self, inter-chain positions use v_cross
        out_self = torch.matmul(attn_weights * intra_mask_float, v_self)
        out_cross = torch.matmul(attn_weights * (1.0 - intra_mask_float), v_cross)
        output = out_self + out_cross

        # Reshape and project
        output = rearrange(output, "b h s d -> b s (h d)")
        output = self.out_proj(output)

        if need_weights:
            return output, attn_weights
        return output


class SharedQKVChainAwareAttention(BaseAttention):
    """
    Chain-aware attention with shared Q/K/V projections.

    Like ChainAwareAttention, this module routes intra-chain queries through
    RoPE-aware scores and inter-chain queries through unrotated (no-position)
    scores, then merges the two score matrices and applies a single global
    softmax. Unlike ChainAwareAttention, it uses one shared set of Q/K/V
    projections (and one shared value path) instead of separate self/cross
    projections.

    HybridNorm is intentionally not supported in the first implementation; the
    config layer rejects that combination.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        dropout: Attention dropout probability.
        bias: Whether to use bias in projections.
        max_seq_len: Maximum sequence length for RoPE.
        qk_norm: QK normalization type ("none", "norm", or "learned_scale").
        norm_type: Normalization type for qk_norm="norm".
        layer_norm_eps: Epsilon for normalization layers.
        hybrid_norm: Must be False; HybridNorm is not supported here.
        rope_fraction: Fraction of head_dim to apply RoPE to.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 512,
        qk_norm: str = "none",
        norm_type: str = "layernorm",
        layer_norm_eps: float = 1e-6,
        hybrid_norm: bool = False,
        rope_fraction: float = 1.0,
    ) -> None:
        if hybrid_norm:
            raise ValueError(
                "SharedQKVChainAwareAttention does not support HybridNorm. "
                "Set hybrid_norm='none' or use chain_aware_projection_mode='separate'."
            )
        super().__init__(
            d_model, n_heads, head_dim, dropout, bias, max_seq_len,
            qk_norm=qk_norm, norm_type=norm_type, layer_norm_eps=layer_norm_eps,
            hybrid_norm=False,
            rope_fraction=rope_fraction,
        )

        # Shared Q/K/V projections (named like MultiHeadAttention for parity)
        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=bias)

    def _create_chain_mask(self, chain_ids: Tensor) -> Tensor:
        return _create_chain_mask(chain_ids)

    def forward(
        self,
        x: Tensor,
        chain_ids: Tensor,
        attention_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass with shared-QKV chain-aware attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            chain_ids: Chain identity tensor of shape (batch, seq_len).
            attention_mask: Optional padding mask of shape (batch, seq_len).
            need_weights: If True, also return merged attention weights.

        Returns:
            Output tensor of shape (batch, seq_len, d_model), or
            (output, attn_weights) when need_weights=True.
        """
        batch_size, seq_len, _ = x.shape

        # Single shared projection
        q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=self.n_heads)

        # Build RoPE-rotated and unrotated views of Q/K. Note: RoPE acts on
        # the leading rope_fraction dims only when partial RoPE is configured.
        q_rope, k_rope = self.rope(q, k)
        q_nope, k_nope = q, k

        # Apply the shared QK norm to both views (preserves the existing
        # ordering where RoPE is applied before QK norm).
        q_rope, k_rope = self._apply_qk_norm(q_rope, k_rope)
        q_nope, k_nope = self._apply_qk_norm(q_nope, k_nope)

        # Score matrices for intra-chain (RoPE) and inter-chain (no-position)
        scores_intra = torch.matmul(q_rope, k_rope.transpose(-2, -1)) * self.scale
        scores_inter = torch.matmul(q_nope, k_nope.transpose(-2, -1)) * self.scale

        intra_mask = self._create_chain_mask(chain_ids)
        merged_scores = torch.where(intra_mask, scores_intra, scores_inter)

        padding_mask = self._create_padding_mask(attention_mask, x.dtype)
        if padding_mask is not None:
            merged_scores = merged_scores + padding_mask

        attn_weights = F.softmax(merged_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        output = rearrange(output, "b h s d -> b s (h d)")
        output = self.out_proj(output)

        if need_weights:
            return output, attn_weights
        return output
