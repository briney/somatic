"""SomaticConfig — PretrainedConfig subclass carrying every model hyperparameter."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

__all__ = ["SomaticConfig"]


_VALID_NORM_TYPES = ("layernorm", "rmsnorm")
_VALID_QK_NORM_MODES = ("none", "norm", "learned_scale")
_VALID_HYBRID_NORMS = ("none", "standard", "star")
_VALID_GRADIENT_CHECKPOINTING_MODES = ("full", "selective")
_VALID_CHAIN_AWARE_PROJECTION_MODES = ("separate", "shared")
_VALID_CLASSIFIER_POOLS = ("mean", "cls")

_DEFAULT_VOCAB_SIZE = 32


class SomaticConfig(PretrainedConfig):
    """Configuration for the Somatic family of paired-chain antibody language models.

    Maps 1:1 to the `model:` block of the project YAML schema and to the
    constructor kwargs of every `Somatic*` class. Field names follow the
    HuggingFace canonical convention (`hidden_size`, `num_hidden_layers`, ...);
    Somatic-specific knobs (chain-aware attention, HybridNorm, RoPE fraction)
    keep their original names.
    """

    model_type = "somatic"

    def __init__(
        self,
        *,
        vocab_size: int = _DEFAULT_VOCAB_SIZE,
        hidden_size: int = 256,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 4,
        head_dim: int | None = None,
        intermediate_size: int | None = None,
        ffn_multiplier: float | None = None,
        max_position_embeddings: int = 320,
        rope_fraction: float = 1.0,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_chain_aware_attention: bool = True,
        chain_aware_projection_mode: str = "separate",
        norm_type: str = "layernorm",
        pre_norm: bool = True,
        post_norm: bool = False,
        qk_norm: str = "none",
        norm_eps: float = 1e-6,
        hybrid_norm: str = "none",
        gradient_checkpointing: bool = False,
        gradient_checkpointing_mode: str = "full",
        initializer_range: float = 0.02,
        classifier_pool: str = "mean",
        classifier_dropout: float = 0.0,
        num_labels: int = 2,
        pre_head_norm: bool = False,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        mask_token_id: int = 31,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.head_dim = head_dim if head_dim is None else int(head_dim)
        self.intermediate_size = (
            intermediate_size if intermediate_size is None else int(intermediate_size)
        )
        self.ffn_multiplier = ffn_multiplier if ffn_multiplier is None else float(ffn_multiplier)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_fraction = float(rope_fraction)
        self.hidden_dropout = float(hidden_dropout)
        self.attention_dropout = float(attention_dropout)
        self.use_chain_aware_attention = bool(use_chain_aware_attention)
        self.chain_aware_projection_mode = chain_aware_projection_mode
        self.norm_type = norm_type
        self.pre_norm = bool(pre_norm)
        self.post_norm = bool(post_norm)
        self.qk_norm = qk_norm
        self.norm_eps = float(norm_eps)
        self.hybrid_norm = hybrid_norm
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.gradient_checkpointing_mode = str(gradient_checkpointing_mode)
        self.initializer_range = float(initializer_range)
        self.classifier_pool = classifier_pool
        self.classifier_dropout = float(classifier_dropout)
        # `num_labels` is a property on PretrainedConfig backed by id2label/label2id;
        # forward it through kwargs in super().__init__, never assign it here.
        self.pre_head_norm = bool(pre_head_norm)
        self.unk_token_id = int(unk_token_id)
        self.mask_token_id = int(mask_token_id)

        self._resolve_derived_fields()
        self._validate()

        kwargs.setdefault("num_labels", int(num_labels))

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Derived-field resolution
    # ------------------------------------------------------------------

    def _resolve_derived_fields(self) -> None:
        """Fill in any `None` derived fields from the source dimensions."""
        if self.head_dim is None:
            if self.num_attention_heads == 0:
                raise ValueError("num_attention_heads must be > 0.")
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.ffn_multiplier is None:
            self.ffn_multiplier = 8 / 3

        if self.intermediate_size is None:
            # ~8/3 * D rounded up to a hardware-friendly multiple of 64.
            self.intermediate_size = int(self.hidden_size * self.ffn_multiplier)
            self.intermediate_size = ((self.intermediate_size + 63) // 64) * 64

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Raise `ValueError` on any field combination the model cannot handle."""
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be > 0; got {self.num_attention_heads}.")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads "
                f"({self.num_attention_heads})."
            )
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"head_dim ({self.head_dim}) * num_attention_heads "
                f"({self.num_attention_heads}) must equal hidden_size ({self.hidden_size})."
            )

        if not 0.0 <= self.rope_fraction <= 1.0:
            raise ValueError(f"rope_fraction must be in [0.0, 1.0]; got {self.rope_fraction}.")

        if self.norm_type not in _VALID_NORM_TYPES:
            raise ValueError(
                f"norm_type must be one of {_VALID_NORM_TYPES}; got {self.norm_type!r}."
            )
        if self.hybrid_norm not in _VALID_HYBRID_NORMS:
            raise ValueError(
                f"hybrid_norm must be one of {_VALID_HYBRID_NORMS}; got {self.hybrid_norm!r}."
            )
        if self.gradient_checkpointing_mode not in _VALID_GRADIENT_CHECKPOINTING_MODES:
            raise ValueError(
                f"gradient_checkpointing_mode must be one of "
                f"{_VALID_GRADIENT_CHECKPOINTING_MODES}; got {self.gradient_checkpointing_mode!r}."
            )
        if self.chain_aware_projection_mode not in _VALID_CHAIN_AWARE_PROJECTION_MODES:
            raise ValueError(
                f"chain_aware_projection_mode must be one of "
                f"{_VALID_CHAIN_AWARE_PROJECTION_MODES}; "
                f"got {self.chain_aware_projection_mode!r}."
            )
        if self.classifier_pool not in _VALID_CLASSIFIER_POOLS:
            raise ValueError(
                f"classifier_pool must be one of {_VALID_CLASSIFIER_POOLS}; "
                f"got {self.classifier_pool!r}."
            )

        # Shared-QKV chain-aware attention does not yet support HybridNorm.
        if (
            self.use_chain_aware_attention
            and self.chain_aware_projection_mode == "shared"
            and self.hybrid_norm != "none"
        ):
            raise ValueError(
                "chain_aware_projection_mode='shared' is not compatible with "
                f"hybrid_norm='{self.hybrid_norm}'. Use hybrid_norm='none' or "
                "chain_aware_projection_mode='separate'."
            )

        # pre_norm/post_norm/qk_norm are inert when hybrid_norm is enabled, so
        # only validate them when HybridNorm is disabled.
        if self.hybrid_norm == "none":
            if not self.pre_norm and not self.post_norm:
                raise ValueError("At least one of pre_norm or post_norm must be True.")
            if self.qk_norm not in _VALID_QK_NORM_MODES:
                raise ValueError(
                    f"qk_norm must be one of {_VALID_QK_NORM_MODES}; got {self.qk_norm!r}."
                )
