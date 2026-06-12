"""Transformer block with configurable attention and normalization."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils import checkpoint as torch_checkpoint

from .attention import (
    ChainAwareAttention,
    MultiHeadAttention,
    SharedQKVChainAwareAttention,
)
from .ffn import FusedSwiGLUFFN
from .normalization import create_norm_layer

# CheckpointPolicy is public in torch.utils.checkpoint from torch>=2.4; guard the
# import so this module still loads on older torch (the selective branch in
# TransformerBlock.forward raises a clear error if SAC is then requested).
try:
    from torch.utils.checkpoint import CheckpointPolicy as _CheckpointPolicy
except ImportError:  # torch < 2.4
    # None fallback when CheckpointPolicy is absent (torch < 2.4).
    _CheckpointPolicy = None  # type: ignore[assignment, misc]  # ty: ignore[invalid-assignment]


def _build_sac_save_ops() -> frozenset:
    """Collect the aten overloads selective activation checkpointing keeps resident.

    These are the FLOP-heavy ops that are expensive to recompute but cheap to store:
    matmuls (the chain-aware attention path is matmul-based via ``torch.matmul``; the
    weight-tied projections and lm_head are mm/addmm) and the SDPA backends (used by
    the ``MultiHeadAttention`` fallback). Built through getattr so a missing SDPA
    overload on an older torch simply drops out of the set rather than raising at
    import time. Everything not in this set is recomputed on backward.
    """
    aten = torch.ops.aten
    candidates = (
        "mm",  # Linear (2D) / matmuls
        "addmm",  # Linear with bias
        "bmm",  # batched matmul (attention scores / values)
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_attention_math",  # CPU / math backend
    )
    ops = set()
    for name in candidates:
        overload_packet = getattr(aten, name, None)
        if overload_packet is not None:
            ops.add(overload_packet.default)
    return frozenset(ops)


_SAC_SAVE_OPS = _build_sac_save_ops()


def _sac_policy_fn(ctx, op, *args, **kwargs):
    """SAC policy: keep matmul/SDPA outputs, recompute everything else.

    Defined at module level (not a per-call closure) so torch.compile/Dynamo can
    trace it as a constant global when composed with the checkpoint higher-order
    op — a nested closure here trips ``AsPythonConstantNotImplementedError``.
    """
    if op in _SAC_SAVE_OPS:
        return _CheckpointPolicy.MUST_SAVE
    return _CheckpointPolicy.PREFER_RECOMPUTE


class TransformerBlock(nn.Module):
    """
    Transformer block with configurable attention and normalization strategy.

    The ``norm_strategy`` selects where normalization sits relative to each
    sublayer and the residual:

    - ``"pre"``: pre-norm, ``x = x + Sublayer(Norm(x))``.
    - ``"sandwich"``: Sandwich-LN (Ding et al., arXiv:2105.13290),
      ``x = x + Norm(Sublayer(Norm(x)))`` — a norm on the sublayer input and on
      its output, both outside the residual stream.
    - ``"hybrid"``: HybridNorm (Zhuo et al., arXiv:2503.04598) — QKV-norm inside
      attention with no outer attention pre-norm, and ``Norm(h)`` reused as both
      the FFN input and the FFN-side residual base.

    Either LayerNorm or RMSNorm is used throughout, per ``norm_type``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int = 64,
        d_ffn: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_seq_len: int = 512,
        layer_norm_eps: float = 1e-6,
        use_chain_aware_attention: bool = True,
        chain_aware_projection_mode: str = "separate",
        norm_type: str = "layernorm",
        norm_strategy: str = "pre",
        qk_norm: str = "none",
        rope_fraction: float = 1.0,
    ) -> None:
        super().__init__()

        # Gradient (activation) checkpointing flags. Off by default; the encoder
        # flips these via set_gradient_checkpointing(). Only fires in training mode.
        self.gradient_checkpointing = False
        self.gradient_checkpointing_mode = "full"  # "full" | "selective"

        if norm_strategy not in ("pre", "hybrid", "sandwich"):
            raise ValueError(
                f"Unknown norm_strategy {norm_strategy!r}; "
                "expected one of 'pre', 'hybrid', 'sandwich'."
            )
        self.norm_strategy = norm_strategy
        hybrid = norm_strategy == "hybrid"

        # Attention pre-norm: present under every strategy except hybrid (which
        # normalizes Q/K/V inside attention instead of the attention input).
        self.attn_norm: nn.Module | None = None
        if not hybrid:
            self.attn_norm = create_norm_layer(norm_type, d_model, layer_norm_eps)

        # FFN pre-norm: always present.
        self.ffn_norm = create_norm_layer(norm_type, d_model, layer_norm_eps)

        # Sandwich post-norms: applied to each sublayer output, outside the residual.
        self.attn_post_norm: nn.Module | None = None
        self.ffn_post_norm: nn.Module | None = None
        if norm_strategy == "sandwich":
            self.attn_post_norm = create_norm_layer(norm_type, d_model, layer_norm_eps)
            self.ffn_post_norm = create_norm_layer(norm_type, d_model, layer_norm_eps)

        # Select attention type based on config
        if not use_chain_aware_attention:
            attention_cls = MultiHeadAttention
        elif chain_aware_projection_mode == "separate":
            attention_cls = ChainAwareAttention
        elif chain_aware_projection_mode == "shared":
            attention_cls = SharedQKVChainAwareAttention
        else:
            raise ValueError(
                f"Unknown chain_aware_projection_mode: '{chain_aware_projection_mode}'"
            )
        self.attention = attention_cls(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=attention_dropout,
            max_seq_len=max_seq_len,
            qk_norm=qk_norm,
            norm_type=norm_type,
            layer_norm_eps=layer_norm_eps,
            hybrid_norm=hybrid,
            rope_fraction=rope_fraction,
        )

        assert d_ffn is not None, "d_ffn must be resolved to a concrete size"
        self.ffn = FusedSwiGLUFFN(d_model=d_model, d_ffn=d_ffn, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Dispatch to the block computation, optionally activation-checkpointed.

        Checkpointing only fires in training mode (``self.training``). During
        training ``output_attentions`` is False, so the checkpointed call returns a
        single tensor. ``use_reentrant=False`` is required for torch.compile
        compatibility.
        """
        if self.gradient_checkpointing and self.training:
            if self.gradient_checkpointing_mode == "selective":
                if _CheckpointPolicy is None:
                    raise RuntimeError(
                        "Selective activation checkpointing requires torch>=2.4; "
                        "set gradient_checkpointing_mode='full'."
                    )
                context_fn = functools.partial(
                    torch_checkpoint.create_selective_checkpoint_contexts,
                    _sac_policy_fn,
                )
                return torch_checkpoint.checkpoint(
                    self._forward_impl,
                    x,
                    token_type_ids,
                    attention_mask,
                    output_attentions,
                    use_reentrant=False,
                    context_fn=context_fn,
                )
            # "full" — recompute the entire block on backward.
            return torch_checkpoint.checkpoint(
                self._forward_impl,
                x,
                token_type_ids,
                attention_mask,
                output_attentions,
                use_reentrant=False,
            )
        return self._forward_impl(x, token_type_ids, attention_mask, output_attentions)

    def _forward_impl(
        self,
        x: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_type_ids: Chain identity tensor of shape (batch, seq_len)
            attention_mask: Optional padding mask of shape (batch, seq_len)
            output_attentions: If True, return attention weights

        Returns:
            If output_attentions is False:
                Output tensor of shape (batch, seq_len, d_model)
            If output_attentions is True:
                Tuple of (output, attn_weights) where attn_weights has shape
                (batch, n_heads, seq_len, seq_len)
        """
        # Attention sublayer. Hybrid feeds the raw residual stream (QKV-norm lives
        # inside the attention module); pre/sandwich apply the outer pre-norm.
        if self.norm_strategy == "hybrid":
            attn_base = x
        else:
            assert self.attn_norm is not None
            attn_base = self.attn_norm(x)

        if output_attentions:
            attn_out, attn_weights = self.attention(
                attn_base, token_type_ids, attention_mask, need_weights=True
            )
        else:
            attn_out = self.attention(attn_base, token_type_ids, attention_mask, need_weights=False)

        # Sandwich: post-norm the attention output before the residual add.
        if self.norm_strategy == "sandwich":
            assert self.attn_post_norm is not None
            attn_out = self.attn_post_norm(attn_out)
        h = x + self.dropout(attn_out)

        # FFN sublayer.
        h_norm = self.ffn_norm(h)
        ffn_out = self.ffn(h_norm)
        if self.norm_strategy == "sandwich":
            assert self.ffn_post_norm is not None
            ffn_out = self.ffn_post_norm(ffn_out)
            x = h + self.dropout(ffn_out)
        elif self.norm_strategy == "hybrid":
            # Norm(h) is reused as both the FFN input and the FFN-side residual base.
            x = h_norm + self.dropout(ffn_out)
        else:  # "pre"
            x = h + self.dropout(ffn_out)

        if output_attentions:
            return x, attn_weights
        return x


# Backward compatibility alias
PreNormBlock = TransformerBlock


class TransformerEncoder(nn.Module):
    """Stack of transformer blocks with configurable normalization."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        head_dim: int = 64,
        d_ffn: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_seq_len: int = 512,
        use_chain_aware_attention: bool = True,
        chain_aware_projection_mode: str = "separate",
        norm_type: str = "layernorm",
        norm_strategy: str = "pre",
        qk_norm: str = "none",
        layer_norm_eps: float = 1e-6,
        rope_fraction: float = 1.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    d_ffn=d_ffn,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    max_seq_len=max_seq_len,
                    layer_norm_eps=layer_norm_eps,
                    use_chain_aware_attention=use_chain_aware_attention,
                    chain_aware_projection_mode=chain_aware_projection_mode,
                    norm_type=norm_type,
                    norm_strategy=norm_strategy,
                    qk_norm=qk_norm,
                    rope_fraction=rope_fraction,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = create_norm_layer(norm_type, d_model, layer_norm_eps)

    def set_gradient_checkpointing(self, enabled: bool, mode: str | None = None) -> None:
        """Toggle gradient (activation) checkpointing on every block in the stack.

        Args:
            enabled: Whether activation checkpointing fires during training.
            mode: Optional "full" | "selective" flavor. When given, it is propagated
                to every block; when None, each block's existing mode is left as-is.
        """
        for block in self.layers:
            assert isinstance(block, TransformerBlock)
            block.gradient_checkpointing = enabled
            if mode is not None:
                block.gradient_checkpointing_mode = mode

    def forward(
        self,
        x: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> (
        Tensor
        | tuple[Tensor, tuple[Tensor, ...]]
        | tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]
    ):
        """
        Forward pass through the transformer encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_type_ids: Chain identity tensor of shape (batch, seq_len)
            attention_mask: Optional padding mask of shape (batch, seq_len)
            output_hidden_states: If True, return all hidden states (including input)
            output_attentions: If True, return attention weights from all layers

        Returns:
            If neither output_hidden_states nor output_attentions:
                Output tensor of shape (batch, seq_len, d_model)
            If output_hidden_states only:
                Tuple of (output, hidden_states) where hidden_states is a tuple of
                n_layers + 1 tensors (input embedding + each layer output before final norm)
            If output_attentions only:
                Tuple of (output, attentions) where attentions is a tuple of
                n_layers attention weight tensors
            If both:
                Tuple of (output, hidden_states, attentions)
        """
        all_hidden_states: tuple[Tensor, ...] = ()
        all_attentions: tuple[Tensor, ...] = ()

        # Include input embeddings in hidden states
        if output_hidden_states:
            all_hidden_states = (x,)

        for layer in self.layers:
            if output_attentions:
                x, attn_weights = layer(x, token_type_ids, attention_mask, output_attentions=True)
                all_attentions = all_attentions + (attn_weights,)
            else:
                x = layer(x, token_type_ids, attention_mask, output_attentions=False)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

        x = self.final_norm(x)

        # Build return value based on what was requested
        if output_hidden_states and output_attentions:
            return x, all_hidden_states, all_attentions
        elif output_hidden_states:
            return x, all_hidden_states
        elif output_attentions:
            return x, all_attentions
        return x
