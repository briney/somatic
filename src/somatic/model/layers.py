"""Transformer block with configurable attention and normalization."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .attention import ChainAwareAttention, MultiHeadAttention
from .ffn import FusedSwiGLUFFN
from .normalization import create_norm_layer


class TransformerBlock(nn.Module):
    """
    Transformer block with configurable attention, normalization type, and placement.

    Supports:
    - Pre-norm: x = x + Sublayer(Norm(x))
    - Post-norm: x = Norm(x + Sublayer(x))
    - Both: x = Norm(x + Sublayer(Norm(x)))
    - LayerNorm or RMSNorm
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
        norm_type: str = "layernorm",
        pre_norm: bool = True,
        post_norm: bool = False,
        qk_norm: str = "none",
    ) -> None:
        super().__init__()

        self.pre_norm = pre_norm
        self.post_norm = post_norm

        # Pre-normalization layers (optional)
        self.attention_pre_norm: nn.Module | None = None
        self.ffn_pre_norm: nn.Module | None = None
        if pre_norm:
            self.attention_pre_norm = create_norm_layer(norm_type, d_model, layer_norm_eps)
            self.ffn_pre_norm = create_norm_layer(norm_type, d_model, layer_norm_eps)

        # Post-normalization layers (optional)
        self.attention_post_norm: nn.Module | None = None
        self.ffn_post_norm: nn.Module | None = None
        if post_norm:
            self.attention_post_norm = create_norm_layer(norm_type, d_model, layer_norm_eps)
            self.ffn_post_norm = create_norm_layer(norm_type, d_model, layer_norm_eps)

        # Select attention type based on config
        attention_cls = ChainAwareAttention if use_chain_aware_attention else MultiHeadAttention
        self.attention = attention_cls(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=attention_dropout,
            max_seq_len=max_seq_len,
            qk_norm=qk_norm,
            norm_type=norm_type,
            layer_norm_eps=layer_norm_eps,
        )

        self.ffn = FusedSwiGLUFFN(d_model=d_model, d_ffn=d_ffn, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        chain_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            chain_ids: Chain identity tensor of shape (batch, seq_len)
            attention_mask: Optional padding mask of shape (batch, seq_len)
            output_attentions: If True, return attention weights

        Returns:
            If output_attentions is False:
                Output tensor of shape (batch, seq_len, d_model)
            If output_attentions is True:
                Tuple of (output, attn_weights) where attn_weights has shape
                (batch, n_heads, seq_len, seq_len)
        """
        # Attention sublayer
        residual = x
        if self.pre_norm and self.attention_pre_norm is not None:
            x = self.attention_pre_norm(x)

        if output_attentions:
            attn_out, attn_weights = self.attention(
                x, chain_ids, attention_mask, need_weights=True
            )
        else:
            attn_out = self.attention(x, chain_ids, attention_mask, need_weights=False)

        x = residual + self.dropout(attn_out)

        if self.post_norm and self.attention_post_norm is not None:
            x = self.attention_post_norm(x)

        # FFN sublayer
        residual = x
        if self.pre_norm and self.ffn_pre_norm is not None:
            x = self.ffn_pre_norm(x)

        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)

        if self.post_norm and self.ffn_post_norm is not None:
            x = self.ffn_post_norm(x)

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
        norm_type: str = "layernorm",
        pre_norm: bool = True,
        post_norm: bool = False,
        qk_norm: str = "none",
        layer_norm_eps: float = 1e-6,
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
                    norm_type=norm_type,
                    pre_norm=pre_norm,
                    post_norm=post_norm,
                    qk_norm=qk_norm,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = create_norm_layer(norm_type, d_model, layer_norm_eps)

    def forward(
        self,
        x: Tensor,
        chain_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]] | tuple[
        Tensor, tuple[Tensor, ...], tuple[Tensor, ...]
    ]:
        """
        Forward pass through the transformer encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            chain_ids: Chain identity tensor of shape (batch, seq_len)
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
                x, attn_weights = layer(
                    x, chain_ids, attention_mask, output_attentions=True
                )
                all_attentions = all_attentions + (attn_weights,)
            else:
                x = layer(x, chain_ids, attention_mask, output_attentions=False)

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
