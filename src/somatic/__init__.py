"""Somatic: Antibody Language Model."""

from __future__ import annotations

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from .encoding import SomaticEncoder
from .model import (
    SomaticConfig,
    SomaticForMaskedLM,
    SomaticForSequenceClassification,
    SomaticForTokenClassification,
    SomaticModel,
    SomaticTokenizerFast,
)
from .tokenizer import AA_END_IDX, AA_START_IDX, DEFAULT_VOCAB, Tokenizer, tokenizer
from .version import __version__

# (1) In-process registration so `import somatic` + AutoModel*.from_pretrained
# works without trust_remote_code. HF's `register` raises on a duplicate
# model_type, so guard each call with exist_ok=True to keep re-imports idempotent.
AutoConfig.register("somatic", SomaticConfig, exist_ok=True)
AutoModel.register(SomaticConfig, SomaticModel, exist_ok=True)
AutoModelForMaskedLM.register(SomaticConfig, SomaticForMaskedLM, exist_ok=True)
AutoModelForSequenceClassification.register(
    SomaticConfig, SomaticForSequenceClassification, exist_ok=True
)
AutoModelForTokenClassification.register(
    SomaticConfig, SomaticForTokenClassification, exist_ok=True
)
AutoTokenizer.register(SomaticConfig, fast_tokenizer_class=SomaticTokenizerFast, exist_ok=True)

# (2) Tell HF to copy the custom-code .py files on save_pretrained / push_to_hub
# and write the matching auto_map entries into config.json / tokenizer_config.json.
# Setting auto_map manually is NOT sufficient — register_for_auto_class is the
# documented hook for the file-copy step. These set a class attribute, so repeat
# calls are no-ops.
SomaticConfig.register_for_auto_class("AutoConfig")
SomaticModel.register_for_auto_class("AutoModel")
SomaticForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")
SomaticForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
SomaticForTokenClassification.register_for_auto_class("AutoModelForTokenClassification")
SomaticTokenizerFast.register_for_auto_class("AutoTokenizer")

__all__ = [
    "SomaticModel",
    "SomaticConfig",
    "SomaticForMaskedLM",
    "SomaticForSequenceClassification",
    "SomaticForTokenClassification",
    "SomaticTokenizerFast",
    "SomaticEncoder",
    "Tokenizer",
    "tokenizer",
    "DEFAULT_VOCAB",
    "AA_START_IDX",
    "AA_END_IDX",
    "__version__",
]
