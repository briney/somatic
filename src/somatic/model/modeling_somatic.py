"""All public Somatic* model classes — single file for trust_remote_code compatibility.

Every public model class lives in this one module so that `auto_map` +
`trust_remote_code` loading works without chasing imports across files. Internal
building blocks live in their own modules and are imported here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

# Several of these relative imports (`attention`, `ffn`, `rope`) are not used
# directly here — they are reached transitively via `layers`. They are imported
# anyway so that loading custom code from a local directory with
# trust_remote_code=True bundles every helper file: HuggingFace copies only a
# module's *direct* relative imports (depth-1; see
# transformers.dynamic_module_utils.get_cached_module_file). `_REMOTE_CODE_DEPS`
# references the otherwise-unused names so linters don't flag them.
from .attention import (
    ChainAwareAttention,
    MultiHeadAttention,
    SharedQKVChainAwareAttention,
)
from .configuration_somatic import _VALID_GRADIENT_CHECKPOINTING_MODES, SomaticConfig
from .embeddings import SomaticEmbedding
from .ffn import FusedSwiGLUFFN
from .layers import TransformerBlock, TransformerEncoder
from .normalization import RMSNorm, create_norm_layer
from .rope import RotaryPositionEmbedding

_REMOTE_CODE_DEPS = (
    ChainAwareAttention,
    MultiHeadAttention,
    SharedQKVChainAwareAttention,
    FusedSwiGLUFFN,
    RotaryPositionEmbedding,
    TransformerBlock,
)

__all__ = [
    "SomaticPreTrainedModel",
    "SomaticModel",
    "SomaticForMaskedLM",
    "SomaticForSequenceClassification",
    "SomaticForTokenClassification",
]


def mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Mask-aware mean over the sequence dimension.

    Args:
        hidden: ``(B, T, D)`` tensor.
        attention_mask: ``(B, T)`` tensor with ``1`` at real tokens and ``0`` at
            pad positions.

    Returns:
        ``(B, D)`` tensor. Rows whose mask is entirely zero come back as zeros
        (the denominator is clamped to at least ``1``).
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = attention_mask.sum(dim=1, keepdim=True).to(hidden.dtype).clamp(min=1)
    return summed / counts


def cls_pool(hidden: Tensor) -> Tensor:
    """Return the ``<cls>`` position (token index 0) of every row.

    Args:
        hidden: ``(B, T, D)`` tensor.

    Returns:
        ``(B, D)`` tensor.
    """
    return hidden[:, 0, :]


class SomaticPreTrainedModel(PreTrainedModel):
    """Abstract base for every Somatic model: weight init + gradient checkpointing."""

    config_class = SomaticConfig
    base_model_prefix = "somatic"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using ``config.initializer_range`` as the std."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ------------------------------------------------------------------
    # Gradient checkpointing — propagate the toggle to every block.
    # ------------------------------------------------------------------

    def _set_gradient_checkpointing(self, value: bool, mode: str | None = None) -> None:
        self.gradient_checkpointing = value
        for module in self.modules():
            if isinstance(module, TransformerEncoder):
                module.set_gradient_checkpointing(value, mode)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        """Enable activation checkpointing on every transformer block.

        Honors an optional ``{"mode": "full" | "selective"}`` entry in
        ``gradient_checkpointing_kwargs``; when absent, falls back to
        ``config.gradient_checkpointing_mode`` (default ``"full"``).
        """
        mode = (gradient_checkpointing_kwargs or {}).get("mode")
        if mode is None:
            mode = getattr(self.config, "gradient_checkpointing_mode", "full")
        if mode not in _VALID_GRADIENT_CHECKPOINTING_MODES:
            raise ValueError(
                f"gradient_checkpointing mode must be one of "
                f"{_VALID_GRADIENT_CHECKPOINTING_MODES}; got {mode!r}."
            )
        self._set_gradient_checkpointing(True, mode=mode)

    def gradient_checkpointing_disable(self) -> None:
        """Disable activation checkpointing on every transformer block."""
        self._set_gradient_checkpointing(False)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Total parameter count, optionally excluding the (tied) token embedding."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.get_input_embeddings().weight.numel()
        return n_params


class SomaticModel(SomaticPreTrainedModel):
    """The Somatic encoder backbone — token embedding, N × TransformerBlock, final norm.

    Pre-norm transformer with RoPE, SwiGLU, and chain-aware hybrid self/cross
    attention. Returns a ``BaseModelOutput``; has no task head.
    """

    def __init__(self, config: SomaticConfig) -> None:
        super().__init__(config)

        # `head_dim` / `intermediate_size` are resolved to concrete ints by
        # SomaticConfig; assert narrows the declared Optional types for checkers.
        assert config.head_dim is not None
        assert config.intermediate_size is not None

        self.embeddings = SomaticEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            padding_idx=config.pad_token_id,
            dropout=config.hidden_dropout,
        )

        self.encoder = TransformerEncoder(
            n_layers=config.num_hidden_layers,
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            d_ffn=config.intermediate_size,
            dropout=config.hidden_dropout,
            attention_dropout=config.attention_dropout,
            max_seq_len=config.max_position_embeddings,
            use_chain_aware_attention=config.use_chain_aware_attention,
            chain_aware_projection_mode=config.chain_aware_projection_mode,
            norm_type=config.norm_type,
            pre_norm=config.pre_norm,
            post_norm=config.post_norm,
            qk_norm=config.qk_norm,
            layer_norm_eps=config.norm_eps,
            hybrid_norm=config.hybrid_norm,
            rope_fraction=config.rope_fraction,
        )

        # Apply config-driven activation checkpointing. The dispatch is gated on
        # training mode, so inference bypasses it regardless of this setting.
        self.encoder.set_gradient_checkpointing(
            config.gradient_checkpointing, config.gradient_checkpointing_mode
        )

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.token_embedding.embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embeddings.token_embedding.embedding = value

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> BaseModelOutput | tuple:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Pass exactly one of input_ids or inputs_embeds, not both.")
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            device = inputs_embeds.device
        else:
            raise ValueError("You must provide either input_ids or inputs_embeds.")

        # Default to a single chain (all-zero segment ids) when not supplied.
        if token_type_ids is None:
            token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        embedded = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds

        encoder_outputs = self.encoder(
            embedded,
            token_type_ids,
            attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # The encoder packs its outputs by which flags are set; unpack to a
        # consistent (last_hidden, hidden_states, attentions) triple.
        if output_hidden_states and output_attentions:
            last_hidden, all_hidden_states, all_attentions = encoder_outputs
        elif output_hidden_states:
            last_hidden, all_hidden_states = encoder_outputs
            all_attentions = None
        elif output_attentions:
            last_hidden, all_attentions = encoder_outputs
            all_hidden_states = None
        else:
            last_hidden = encoder_outputs
            all_hidden_states = None
            all_attentions = None

        if not return_dict:
            return tuple(
                v for v in (last_hidden, all_hidden_states, all_attentions) if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=last_hidden,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class SomaticForMaskedLM(SomaticPreTrainedModel):
    """Somatic with a masked-language-model head (bias-free, weight-tied)."""

    _tied_weights_keys = {"lm_head.weight": "somatic.embeddings.token_embedding.embedding.weight"}

    def __init__(self, config: SomaticConfig) -> None:
        super().__init__(config)
        self.somatic = SomaticModel(config)
        # Bias-free decoder; tied to the input embedding via post_init() because
        # config.tie_word_embeddings is True (see _tied_weights_keys).
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.somatic.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.somatic.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> MaskedLMOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.somatic(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            output = tuple(v for v in output if v is not None)
            return ((loss,) + output) if loss is not None else output
        return MaskedLMOutput(
            # transformers stubs annotate loss as FloatTensor; torch ops return Tensor.
            loss=loss,  # ty: ignore[invalid-argument-type]
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def predict_masked(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> Tensor:
        """Fill ``<mask>`` positions with sampled predictions in one forward pass.

        Args:
            input_ids: Input with mask tokens at positions to predict.
            token_type_ids: Chain identity tensor; defaults to a single chain.
            attention_mask: Optional padding mask.
            temperature: Sampling temperature (1.0 = unchanged).
            top_k: If set, restrict sampling to the top-k logits.
            top_p: If set, use nucleus (top-p) sampling.

        Returns:
            ``input_ids`` with masked positions replaced by sampled tokens.
        """
        outputs = self(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None:
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            threshold = top_k_values[..., -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            indices_to_remove = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Special tokens (CLS/PAD/EOS/UNK/MASK) must never be sampled.
        special_tokens = [
            self.config.bos_token_id,
            self.config.pad_token_id,
            self.config.eos_token_id,
            self.config.unk_token_id,
            self.config.mask_token_id,
        ]
        for token_id in special_tokens:
            if token_id is not None:
                logits[..., token_id] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        batch_size, seq_len, vocab_size = probs.shape
        predicted = torch.multinomial(probs.view(-1, vocab_size), 1).view(batch_size, seq_len)

        result = input_ids.clone()
        mask_positions = input_ids == self.config.mask_token_id
        result[mask_positions] = predicted[mask_positions]
        return result


class SomaticForSequenceClassification(SomaticPreTrainedModel):
    """Somatic with a pooled sequence-classification head."""

    def __init__(self, config: SomaticConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.somatic = SomaticModel(config)
        if config.pre_head_norm:
            self.pre_head_norm: nn.Module = create_norm_layer(
                config.norm_type, config.hidden_size, config.norm_eps
            )
        else:
            self.pre_head_norm = nn.Identity()
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.somatic.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.somatic.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> SequenceClassifierOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.somatic(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state

        if self.config.classifier_pool == "cls":
            pooled = cls_pool(last_hidden)
        else:
            pooled_mask = self._pooling_mask(attention_mask, last_hidden)
            pooled = mean_pool(last_hidden, pooled_mask)

        pooled = self.pre_head_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).float()

        loss = self._compute_loss(logits, labels)

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            output = tuple(v for v in output if v is not None)
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            # transformers stubs annotate loss as FloatTensor; torch ops return Tensor.
            loss=loss,  # ty: ignore[invalid-argument-type]
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def _pooling_mask(attention_mask: Tensor | None, hidden: Tensor) -> Tensor:
        if attention_mask is not None:
            return attention_mask
        return hidden.new_ones(hidden.shape[:2], dtype=torch.long)

    def _compute_loss(self, logits: Tensor, labels: Tensor | None) -> Tensor | None:
        """Loss with HF-style ``problem_type`` inference (regression / classification)."""
        if labels is None:
            return None

        problem_type = self.config.problem_type
        if problem_type is None:
            if self.num_labels == 1:
                problem_type = "regression"
            elif labels.dtype in (torch.long, torch.int):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
            self.config.problem_type = problem_type

        if problem_type == "regression":
            if self.num_labels == 1:
                return F.mse_loss(logits.squeeze(-1), labels.squeeze(-1))
            return F.mse_loss(logits, labels)
        if problem_type == "single_label_classification":
            return F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return F.binary_cross_entropy_with_logits(logits, labels.float())


class SomaticForTokenClassification(SomaticPreTrainedModel):
    """Somatic with a per-token classification head."""

    def __init__(self, config: SomaticConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.somatic = SomaticModel(config)
        if config.pre_head_norm:
            self.pre_head_norm: nn.Module = create_norm_layer(
                config.norm_type, config.hidden_size, config.norm_eps
            )
        else:
            self.pre_head_norm = nn.Identity()
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.somatic.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.somatic.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> TokenClassifierOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.somatic(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = self.pre_head_norm(outputs.last_hidden_state)
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden).float()

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            output = tuple(v for v in output if v is not None)
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            # transformers stubs annotate loss as FloatTensor; torch ops return Tensor.
            loss=loss,  # ty: ignore[invalid-argument-type]
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
