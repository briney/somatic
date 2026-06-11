"""Tokenizer for paired antibody sequences (HuggingFace-compatible)."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from tokenizers import Regex
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "SomaticTokenizerFast",
    "tokenizer",
    "DEFAULT_VOCAB",
    "AA_START_IDX",
    "AA_END_IDX",
]

# Fixed 32-token vocabulary for antibody sequences (index == token id).
DEFAULT_VOCAB = [
    "<cls>",  # 0: Classification/start token
    "<pad>",  # 1: Padding token
    "<eos>",  # 2: End of sequence token
    "<unk>",  # 3: Unknown token
    "L",  # 4-23: Standard amino acids
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",  # 24-28: Non-standard amino acids
    "B",  # Asparagine or Aspartic acid
    "U",  # Selenocysteine
    "O",  # Pyrrolysine
    "Z",  # Glutamine or Glutamic acid
    ".",  # 29: Insertion marker
    "-",  # 30: Gap marker
    "<mask>",  # 31: Mask token for MLM
]

# Amino acid range (for sampling during generation)
AA_START_IDX = 4
AA_END_IDX = 30  # Exclusive


class SomaticTokenizerFast(PreTrainedTokenizerFast):
    """
    Tokenizer for paired antibody sequences with a fixed 32-token vocabulary.

    Character-level tokenization over amino acids with special-token handling.
    The vocabulary mirrors ESM-2 style but with a size of 32 (a multiple of 8 for
    GPU efficiency).

    The backing ``tokenizers.Tokenizer`` emits ``token_type_ids`` natively via its
    post-processor template, reproducing the chain layout used by chain-aware
    attention: ``<cls>`` and the heavy chain are segment 0; the light chain and
    the trailing ``<eos>`` are segment 1. A single sequence is entirely segment 0.

    Examples
    --------
    >>> tokenizer = SomaticTokenizerFast()
    >>> tokenizer.encode("ACDEF")
    [0, 5, 23, 13, 9, 18, 2]  # <cls> A C D E F <eos>
    >>> out = tokenizer("EVQ", "DIQ", return_token_type_ids=True)
    >>> out["token_type_ids"]
    [0, 0, 0, 0, 1, 1, 1, 1]  # <cls> heavy=0 ... light=1 ... <eos>=1
    """

    vocab_files_names = {"tokenizer_file": "tokenizer.json"}
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str | None = None,
        tokenizer_file: str | None = None,
        bos_token: str = "<cls>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        **kwargs,
    ):
        # Build the backend programmatically only for a fresh tokenizer. On
        # reload, HF supplies a serialized `tokenizer_file` (or `tokenizer_object`)
        # and the special-token kwargs arrive as `AddedToken` objects, so coerce
        # to str before handing them to the backend builders.
        if tokenizer_file is None and kwargs.get("tokenizer_object") is None:
            kwargs["tokenizer_object"] = self._build_backend(
                vocab_file=vocab_file,
                bos_token=str(bos_token),
                eos_token=str(eos_token),
                unk_token=str(unk_token),
            )

        # setdefault (not an explicit arg) so a reload, where this already sits in
        # the serialized kwargs, does not collide with a duplicate keyword.
        kwargs.setdefault("clean_up_tokenization_spaces", False)

        super().__init__(
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

    @staticmethod
    def _build_backend(
        vocab_file: str | None,
        bos_token: str,
        eos_token: str,
        unk_token: str,
    ) -> HFTokenizer:
        """Construct the backing fast ``tokenizers.Tokenizer`` from the vocabulary.

        WordLevel model + character/special-token pre-tokenization + a
        ``TemplateProcessing`` post-processor that wraps with ``<cls>``/``<eos>``
        and emits segment (token_type) ids: single -> all 0; pair -> heavy/cls = 0,
        light/eos = 1.
        """
        if vocab_file is not None and os.path.isfile(vocab_file):
            with open(vocab_file, encoding="utf-8") as f:
                vocab = [line.strip() for line in f if line.strip()]
        else:
            vocab = DEFAULT_VOCAB

        vocab_dict = {token: i for i, token in enumerate(vocab)}

        backend = HFTokenizer(WordLevel(vocab=vocab_dict, unk_token=unk_token))

        special_start_char = Regex(r"[<\[]")
        special_end_char = Regex(r"[>\]]")
        pattern = "|".join(re.escape(tok) for tok in vocab if len(tok) == 1)

        backend.pre_tokenizer = Sequence(
            [
                Split(special_start_char, behavior="merged_with_next"),
                Split(special_end_char, behavior="merged_with_previous"),
                Split(Regex(pattern), behavior="isolated"),
            ]
        )

        backend.post_processor = TemplateProcessing(
            single=f"{bos_token}:0 $A:0 {eos_token}:0",
            pair=f"{bos_token}:0 $A:0 $B:1 {eos_token}:1",
            special_tokens=[
                (bos_token, vocab_dict[bos_token]),
                (eos_token, vocab_dict[eos_token]),
            ],
        )
        return backend

    @property
    def cls_token_id(self) -> int:
        """Get the CLS token ID (alias for bos_token_id)."""
        return self.bos_token_id

    @property
    def cls_token(self) -> str:
        """Get the CLS token (alias for bos_token)."""
        return self.bos_token

    def encode_paired(
        self,
        heavy_chain: str,
        light_chain: str,
        return_tensors: str | None = None,
    ) -> dict[str, list[int]] | dict[str, Tensor]:
        """
        Encode paired heavy/light chain sequences.

        Format: ``<cls> heavy light <eos>`` with segment ids
        ``[0, 0..., 1..., 1]`` (cls + heavy = 0, light + eos = 1). This is a thin
        wrapper over ``self(heavy, light, return_token_type_ids=True)``.

        Parameters
        ----------
        heavy_chain
            Amino acid sequence of the heavy chain.
        light_chain
            Amino acid sequence of the light chain.
        return_tensors
            If ``"pt"``, return batched PyTorch tensors; if ``None``, return lists.

        Returns
        -------
        dict
            Dictionary with ``input_ids``, ``token_type_ids``, ``attention_mask``.

        Examples
        --------
        >>> tokenizer = SomaticTokenizerFast()
        >>> result = tokenizer.encode_paired("AC", "DE")
        >>> result["input_ids"]
        [0, 5, 23, 13, 9, 2]  # <cls> A C D E <eos>
        >>> result["token_type_ids"]
        [0, 0, 0, 1, 1, 1]
        """
        encoded = self(
            heavy_chain,
            light_chain,
            return_token_type_ids=True,
            return_tensors=return_tensors,
        )
        return {
            "input_ids": encoded["input_ids"],
            "token_type_ids": encoded["token_type_ids"],
            "attention_mask": encoded["attention_mask"],
        }


# Module-level singleton for convenience.
tokenizer = SomaticTokenizerFast()
