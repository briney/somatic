"""Main Somatic transformer model."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ..tokenizer import tokenizer
from .embeddings import SomaticEmbedding
from .layers import TransformerEncoder
from .normalization import RMSNorm


@dataclass
class SomaticConfig:
    """Configuration for Somatic model."""

    vocab_size: int = 32
    padding_idx: int = 1  # tokenizer.pad_token_id

    d_model: int = 256
    n_layers: int = 16
    n_heads: int = 4
    d_ffn: int | None = None
    ffn_multiplier: float | None = None  # Default 8/3 when None

    # Deprecated: head_dim is now computed as d_model // n_heads
    head_dim: int | None = None

    max_seq_len: int = 320

    dropout: float = 0.1
    attention_dropout: float = 0.1
    embedding_dropout: float = 0.1

    # If True, use ChainAwareAttention (MINT-style hybrid attention)
    # If False, use standard MultiHeadAttention
    use_chain_aware_attention: bool = True

    # Normalization options
    norm_type: str = "layernorm"  # "layernorm" or "rmsnorm"
    pre_norm: bool = True  # Apply normalization before attention/FFN
    post_norm: bool = False  # Apply normalization after attention/FFN
    qk_norm: str = "none"  # "none", "norm", or "learned_scale"
    layer_norm_eps: float = 1e-6  # Epsilon for normalization layers

    def __post_init__(self) -> None:
        # Validate and compute head_dim
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        computed_head_dim = self.d_model // self.n_heads
        if self.head_dim is not None and self.head_dim != computed_head_dim:
            warnings.warn(
                f"head_dim is deprecated and computed automatically. "
                f"Using {computed_head_dim} (d_model // n_heads).",
                DeprecationWarning,
                stacklevel=2,
            )
        self.head_dim = computed_head_dim

        # Set default ffn_multiplier if not provided
        if self.ffn_multiplier is None:
            self.ffn_multiplier = 8 / 3

        # Compute d_ffn using ffn_multiplier if not provided
        if self.d_ffn is None:
            self.d_ffn = int(self.d_model * self.ffn_multiplier)
            self.d_ffn = ((self.d_ffn + 63) // 64) * 64

        # Validate norm_type
        valid_norm_types = {"layernorm", "rmsnorm"}
        if self.norm_type not in valid_norm_types:
            raise ValueError(
                f"norm_type must be one of {valid_norm_types}, got '{self.norm_type}'"
            )

        # Validate pre_norm/post_norm - at least one must be True
        if not self.pre_norm and not self.post_norm:
            raise ValueError("At least one of pre_norm or post_norm must be True")

        # Validate qk_norm
        valid_qk_norms = {"none", "norm", "learned_scale"}
        if self.qk_norm not in valid_qk_norms:
            raise ValueError(
                f"qk_norm must be one of {valid_qk_norms}, got '{self.qk_norm}'"
            )


class SomaticModel(nn.Module):
    """
    Antibody Language Model with masked language modeling objective.

    Pre-norm transformer with RoPE, SwiGLU, and hybrid self/cross attention.
    """

    def __init__(self, config: SomaticConfig) -> None:
        super().__init__()
        self.config = config

        self.embeddings = SomaticEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            padding_idx=config.padding_idx,
            dropout=config.embedding_dropout,
        )

        self.encoder = TransformerEncoder(
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            d_ffn=config.d_ffn,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            max_seq_len=config.max_seq_len,
            use_chain_aware_attention=config.use_chain_aware_attention,
            norm_type=config.norm_type,
            pre_norm=config.pre_norm,
            post_norm=config.post_norm,
            qk_norm=config.qk_norm,
            layer_norm_eps=config.layer_norm_eps,
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.token_embedding.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        token_ids: Tensor,
        chain_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        """
        Forward pass through the model.

        Args:
            token_ids: Input token IDs of shape (batch, seq_len)
            chain_ids: Chain identity tensor of shape (batch, seq_len)
            attention_mask: Optional padding mask of shape (batch, seq_len)
            output_hidden_states: If True, return all hidden states (n_layers + 1 tensors)
            output_attentions: If True, return attention weights from all layers

        Returns:
            Dictionary with:
                - "logits": Output logits of shape (batch, seq_len, vocab_size)
                - "hidden_states": Final hidden states of shape (batch, seq_len, d_model)
                - "all_hidden_states": (optional) Tuple of n_layers + 1 hidden state tensors
                - "attentions": (optional) Tuple of n_layers attention weight tensors
        """
        embedded = self.embeddings(token_ids)

        # Call encoder with appropriate flags
        encoder_outputs = self.encoder(
            embedded,
            chain_ids,
            attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # Parse encoder outputs based on what was requested
        if output_hidden_states and output_attentions:
            hidden_states, all_hidden_states, all_attentions = encoder_outputs
        elif output_hidden_states:
            hidden_states, all_hidden_states = encoder_outputs
            all_attentions = None
        elif output_attentions:
            hidden_states, all_attentions = encoder_outputs
            all_hidden_states = None
        else:
            hidden_states = encoder_outputs
            all_hidden_states = None
            all_attentions = None

        logits = self.lm_head(hidden_states)

        output = {"logits": logits, "hidden_states": hidden_states}

        if all_hidden_states is not None:
            output["all_hidden_states"] = all_hidden_states

        if all_attentions is not None:
            output["attentions"] = all_attentions

        return output

    @torch.no_grad()
    def predict_masked(
        self,
        token_ids: Tensor,
        chain_ids: Tensor,
        attention_mask: Tensor | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> Tensor:
        """
        Predict tokens at masked positions in a single forward pass.

        Args:
            token_ids: Input with MASK tokens at positions to predict
            chain_ids: Chain identity tensor
            attention_mask: Optional attention mask
            temperature: Sampling temperature (1.0 = no change)
            top_k: If set, only sample from top-k tokens
            top_p: If set, use nucleus sampling

        Returns:
            Token IDs with masked positions filled by predictions
        """
        outputs = self.forward(token_ids, chain_ids, attention_mask)
        logits = outputs["logits"]

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None:
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            threshold = top_k_values[..., -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            indices_to_remove = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Prevent sampling special tokens (CLS, PAD, EOS, UNK, MASK)
        # These should never be valid predictions
        special_tokens = [
            tokenizer.cls_token_id,
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.unk_token_id,
            tokenizer.mask_token_id,
        ]
        for token_id in special_tokens:
            if token_id is not None:
                logits[..., token_id] = float("-inf")

        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        batch_size, seq_len, vocab_size = probs.shape
        predicted = torch.multinomial(probs.view(-1, vocab_size), 1)
        predicted = predicted.view(batch_size, seq_len)

        # Only replace MASK tokens
        mask_token_id = tokenizer.mask_token_id
        result = token_ids.clone()
        mask_positions = token_ids == mask_token_id
        result[mask_positions] = predicted[mask_positions]

        return result

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.token_embedding.embedding.weight.numel()
        return n_params

    @classmethod
    def from_pretrained(cls, path: str, map_location: str = "cpu") -> "SomaticModel":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        if "config" not in checkpoint:
            raise ValueError(
                "Checkpoint missing 'config'. This appears to be a training checkpoint "
                "created before config was included. Use model.save_pretrained() to "
                "create an inference checkpoint, or load manually with:\n\n"
                "    checkpoint = torch.load('path/to/checkpoint.pt')\n"
                "    config = SomaticConfig(...)  # with your model's config\n"
                "    model = SomaticModel(config)\n"
                "    model.load_state_dict(checkpoint['model_state_dict'])"
            )

        config_dict = checkpoint["config"]
        # Handle legacy checkpoints with timestep-related fields
        config_dict.pop("max_timesteps", None)
        config_dict.pop("use_timestep_embedding", None)

        config = SomaticConfig(**config_dict)
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, path: str) -> None:
        torch.save(
            {
                "config": asdict(self.config),
                "model_state_dict": self.state_dict(),
            },
            path,
        )
