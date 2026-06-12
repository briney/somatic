"""Backward-compatible shim.

The tokenizer now lives in :mod:`somatic.model.tokenization_somatic`. This module
re-exports it (and a legacy ``Tokenizer`` alias) so existing imports keep working
while consumers are migrated. Remove once all imports point at the new location.
"""

from __future__ import annotations

from .model.tokenization_somatic import (
    AA_END_IDX,
    AA_START_IDX,
    DEFAULT_VOCAB,
    SomaticTokenizerFast,
    tokenizer,
)

# The class was named `Tokenizer` before the HF-compatibility refactor.
Tokenizer = SomaticTokenizerFast

__all__ = [
    "Tokenizer",
    "SomaticTokenizerFast",
    "tokenizer",
    "DEFAULT_VOCAB",
    "AA_START_IDX",
    "AA_END_IDX",
]
