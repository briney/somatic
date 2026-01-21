"""Tokenizer for antibody sequences."""

from __future__ import annotations

import os
import re

from tokenizers import Regex
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

__all__ = [
    "Tokenizer",
    "tokenizer",
    "DEFAULT_VOCAB",
    "AA_START_IDX",
    "AA_END_IDX",
]

# Fixed 32-token vocabulary for antibody sequences
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


class Tokenizer(PreTrainedTokenizerFast):
    """
    Tokenizer for antibody sequences with a fixed 32-token vocabulary.

    This tokenizer uses character-level tokenization for amino acid sequences
    with special token handling. The vocabulary mirrors ESM-2 style but with
    a vocabulary size of 32 (a multiple of 8 for GPU optimization).

    Parameters
    ----------
    vocab_file
        Path to vocabulary file. If not provided, uses DEFAULT_VOCAB.
    tokenizer_file
        Path to a saved tokenizer file (for loading pretrained).
    bos_token
        Beginning/classification token. Default is "<cls>".
    eos_token
        End of sequence token. Default is "<eos>".
    unk_token
        Unknown token. Default is "<unk>".
    pad_token
        Padding token. Default is "<pad>".
    mask_token
        Mask token for MLM. Default is "<mask>".
    **kwargs
        Additional arguments passed to PreTrainedTokenizerFast.

    Examples
    --------
    >>> tokenizer = Tokenizer()
    >>> tokenizer.encode("ACDEF")
    [0, 5, 23, 13, 9, 18, 2]  # [CLS] A C D E F [EOS]
    >>> tokenizer.decode([0, 5, 23, 13, 9, 18, 2])
    '<cls>ACDEF<eos>'
    """

    model_input_names = ["input_ids", "attention_mask"]

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
        # Load pretrained tokenizer (used by AutoTokenizer)
        if tokenizer_file is not None:
            super().__init__(
                tokenizer_file=tokenizer_file,
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                pad_token=pad_token,
                mask_token=mask_token,
                clean_up_tokenization_spaces=False,
                **kwargs,
            )
            return

        # Parse vocabulary
        if vocab_file is not None and os.path.isfile(vocab_file):
            with open(vocab_file, encoding="utf-8") as f:
                vocab = [line.strip() for line in f if line.strip()]
        else:
            vocab = DEFAULT_VOCAB

        vocab_dict = {token: i for i, token in enumerate(vocab)}

        # Create tokenizer with WordLevel model
        tokenizer = HFTokenizer(
            WordLevel(
                vocab=vocab_dict,
                unk_token=unk_token,
            )
        )

        # Regex patterns for pre-tokenization
        special_start_char = Regex(r"[<\[]")
        special_end_char = Regex(r"[>\]]")

        # Pattern for single-character tokens (amino acids and special chars)
        pattern = "|".join(re.escape(tok) for tok in vocab if len(tok) == 1)

        # Pre-tokenization: split on special tokens and individual characters
        tokenizer.pre_tokenizer = Sequence(
            [
                Split(special_start_char, behavior="merged_with_next"),
                Split(special_end_char, behavior="merged_with_previous"),
                Split(Regex(pattern), behavior="isolated"),
            ]
        )

        # Post-processing: add BOS (cls) and EOS tokens
        tokenizer.post_processor = TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            pair=f"{bos_token} $A $B {eos_token}",
            special_tokens=[
                (bos_token, vocab_dict[bos_token]),
                (eos_token, vocab_dict[eos_token]),
            ],
        )

        # Initialize PreTrainedTokenizerFast
        super().__init__(
            tokenizer_object=tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )

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
        add_chain_separator: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, list[int]]:
        """
        Encode paired heavy/light chain sequences.

        When using ChainAwareAttention (add_chain_separator=False):
            Format: <cls> heavy_chain light_chain <eos>
            Chain IDs: [0, 0, 0, ..., 1, 1, 1, ..., 1]

        When using MultiHeadAttention (add_chain_separator=True):
            Format: <cls> heavy_chain <cls> light_chain <eos>
            Chain IDs: [0, 0, 0, ..., 0, 1, 1, 1, ..., 1]
            The separating <cls> acts as a boundary marker between chains.

        Parameters
        ----------
        heavy_chain
            Amino acid sequence of the heavy chain.
        light_chain
            Amino acid sequence of the light chain.
        add_chain_separator
            If True, add a <cls> token between chains (for MultiHeadAttention).
            If False, no separator is added (for ChainAwareAttention).
        return_tensors
            If "pt", return PyTorch tensors. If None, return lists.

        Returns
        -------
        dict
            Dictionary with:
            - "input_ids": Token IDs
            - "chain_ids": Chain identity (0 for heavy, 1 for light)
            - "attention_mask": Attention mask (all 1s)

        Examples
        --------
        >>> tokenizer = Tokenizer()
        >>> result = tokenizer.encode_paired("AC", "DE", add_chain_separator=False)
        >>> result["input_ids"]
        [0, 5, 23, 13, 9, 2]  # <cls> A C D E <eos>
        >>> result["chain_ids"]
        [0, 0, 0, 1, 1, 1]

        >>> result = tokenizer.encode_paired("AC", "DE", add_chain_separator=True)
        >>> result["input_ids"]
        [0, 5, 23, 0, 13, 9, 2]  # <cls> A C <cls> D E <eos>
        >>> result["chain_ids"]
        [0, 0, 0, 0, 1, 1, 1]
        """
        # Encode each chain without special tokens
        heavy_ids = self.encode(heavy_chain, add_special_tokens=False)
        light_ids = self.encode(light_chain, add_special_tokens=False)

        # Build token sequence
        if add_chain_separator:
            # <cls> heavy <cls> light <eos>
            input_ids = (
                [self.cls_token_id]
                + heavy_ids
                + [self.cls_token_id]
                + light_ids
                + [self.eos_token_id]
            )
            # Chain IDs: CLS and heavy = 0, separator CLS and light and EOS = 1
            chain_ids = (
                [0] * (1 + len(heavy_ids) + 1)  # CLS + heavy + separator CLS
                + [1] * (len(light_ids) + 1)  # light + EOS
            )
        else:
            # <cls> heavy light <eos>
            input_ids = (
                [self.cls_token_id]
                + heavy_ids
                + light_ids
                + [self.eos_token_id]
            )
            # Chain IDs: CLS and heavy = 0, light and EOS = 1
            chain_ids = (
                [0] * (1 + len(heavy_ids))  # CLS + heavy
                + [1] * (len(light_ids) + 1)  # light + EOS
            )

        attention_mask = [1] * len(input_ids)

        result = {
            "input_ids": input_ids,
            "chain_ids": chain_ids,
            "attention_mask": attention_mask,
        }

        if return_tensors == "pt":
            import torch

            result = {k: torch.tensor([v]) for k, v in result.items()}

        return result


# Module-level instance for convenience
tokenizer = Tokenizer()
