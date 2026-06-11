"""Main Somatic transformer model."""

from __future__ import annotations

from dataclasses import asdict
from typing import NotRequired, TypedDict

import torch
import torch.nn as nn
from torch import Tensor

from ..tokenizer import tokenizer
from .configuration_somatic import SomaticConfig
from .embeddings import SomaticEmbedding
from .layers import TransformerEncoder
from .normalization import RMSNorm


class ModelOutput(TypedDict):
    """Structured return type of :meth:`SomaticModel.forward`.

    ``logits`` and ``hidden_states`` are always present; the tuple-valued
    ``all_hidden_states`` and ``attentions`` are only included when the
    corresponding ``output_*`` flag is set.
    """

    logits: Tensor
    hidden_states: Tensor
    all_hidden_states: NotRequired[tuple[Tensor, ...]]
    attentions: NotRequired[tuple[Tensor, ...]]


class SomaticModel(nn.Module):
    """
    Antibody Language Model with masked language modeling objective.

    Pre-norm transformer with RoPE, SwiGLU, and hybrid self/cross attention.
    """

    def __init__(self, config: SomaticConfig) -> None:
        super().__init__()
        self.config = config

        # __post_init__ resolves these to concrete ints; assert narrows the
        # declared Optional types for static checkers.
        assert config.head_dim is not None
        assert config.d_ffn is not None

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
            chain_aware_projection_mode=config.chain_aware_projection_mode,
            norm_type=config.norm_type,
            pre_norm=config.pre_norm,
            post_norm=config.post_norm,
            qk_norm=config.qk_norm,
            layer_norm_eps=config.layer_norm_eps,
            hybrid_norm=config.hybrid_norm,
            rope_fraction=config.rope_fraction,
        )

        # Configure gradient (activation) checkpointing from config. Propagates the
        # on/off flag and mode to every block; the dispatch is gated on training mode,
        # so inference (eval/encode) bypasses checkpointing regardless of this setting.
        self.encoder.set_gradient_checkpointing(
            config.gradient_checkpointing, config.gradient_checkpointing_mode
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
    ) -> ModelOutput:
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

        output: ModelOutput = {"logits": logits, "hidden_states": hidden_states}

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
    def from_pretrained(cls, path: str, map_location: str = "cpu") -> SomaticModel:
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
        # Coerce legacy bool hybrid_norm (pre-string-enum) to its string equivalent
        if isinstance(config_dict.get("hybrid_norm"), bool):
            config_dict["hybrid_norm"] = "standard" if config_dict["hybrid_norm"] else "none"

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
