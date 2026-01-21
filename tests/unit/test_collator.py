"""Tests for batch collation."""

import pytest
import torch

from somatic.data.collator import AntibodyCollator
from somatic.tokenizer import tokenizer


class TestAntibodyCollator:
    @pytest.fixture
    def collator(self):
        return AntibodyCollator(max_length=64)

    @pytest.fixture
    def sample_batch(self):
        return [
            {
                "heavy_chain": "EVQLVESGGGLVQ",
                "light_chain": "DIQMTQSPSSLSA",
                "heavy_cdr_mask": None,
                "light_cdr_mask": None,
                "heavy_non_templated_mask": None,
                "light_non_templated_mask": None,
            },
            {
                "heavy_chain": "QVQLQQSGAELA",
                "light_chain": "DIVMTQSPDSLAV",
                "heavy_cdr_mask": None,
                "light_cdr_mask": None,
                "heavy_non_templated_mask": None,
                "light_non_templated_mask": None,
            },
        ]

    def test_output_keys(self, collator, sample_batch):
        result = collator(sample_batch)

        assert "token_ids" in result
        assert "chain_ids" in result
        assert "attention_mask" in result
        assert "special_tokens_mask" in result

    def test_output_shapes(self, collator, sample_batch):
        result = collator(sample_batch)
        batch_size = len(sample_batch)

        # Sequence length should be max of encoded lengths
        seq_len = result["token_ids"].shape[1]

        assert result["token_ids"].shape == (batch_size, seq_len)
        assert result["chain_ids"].shape == (batch_size, seq_len)
        assert result["attention_mask"].shape == (batch_size, seq_len)

    def test_cls_and_eos_tokens(self, collator, sample_batch):
        result = collator(sample_batch)

        # First token should be CLS
        assert (result["token_ids"][:, 0] == tokenizer.cls_token_id).all()

        # Should have EOS somewhere in the sequence
        has_eos = (result["token_ids"] == tokenizer.eos_token_id).any(dim=1)
        assert has_eos.all()

    def test_chain_ids(self, collator, sample_batch):
        result = collator(sample_batch)

        # Chain IDs should be 0 for heavy, 1 for light
        # First token (CLS) should be chain 0
        assert (result["chain_ids"][:, 0] == 0).all()

    def test_attention_mask(self, collator, sample_batch):
        result = collator(sample_batch)

        # Attention mask should be 1 for real tokens, 0 for padding
        for i in range(len(sample_batch)):
            # Find where padding starts
            padding_start = (result["token_ids"][i] == tokenizer.pad_token_id).nonzero()
            if len(padding_start) > 0:
                pad_idx = padding_start[0].item()
                assert result["attention_mask"][i, :pad_idx].sum() == pad_idx
                assert result["attention_mask"][i, pad_idx:].sum() == 0

    def test_special_tokens_mask(self, collator, sample_batch):
        result = collator(sample_batch)

        # CLS (position 0) should be marked as special
        assert result["special_tokens_mask"][:, 0].all()

    def test_pad_to_max(self):
        collator = AntibodyCollator(max_length=100, pad_to_max=True)
        batch = [
            {
                "heavy_chain": "EVQL",
                "light_chain": "DIQM",
                "heavy_cdr_mask": None,
                "light_cdr_mask": None,
                "heavy_non_templated_mask": None,
                "light_non_templated_mask": None,
            }
        ]

        result = collator(batch)
        assert result["token_ids"].shape[1] == 100

    def test_with_cdr_masks(self):
        """Test collation with detailed CDR masks (0=FW, 1=CDR1, 2=CDR2, 3=CDR3)."""
        collator = AntibodyCollator(max_length=64)
        batch = [
            {
                "heavy_chain": "EVQLVE",
                "light_chain": "DIQMTQ",
                # Detailed CDR mask: 0=FW, 1=CDR1, 2=CDR2, 3=CDR3
                "heavy_cdr_mask": [0, 0, 1, 1, 2, 2],
                "light_cdr_mask": [0, 3, 3, 0, 0, 0],
                "heavy_non_templated_mask": None,
                "light_non_templated_mask": None,
            }
        ]

        result = collator(batch)
        assert result["cdr_mask"] is not None
        assert result["cdr_mask"].shape == result["token_ids"].shape
        # Detailed values should be preserved
        cdr_values = result["cdr_mask"][0].tolist()
        assert 1 in cdr_values  # CDR1
        assert 2 in cdr_values  # CDR2
        assert 3 in cdr_values  # CDR3

    def test_truncation(self):
        collator = AntibodyCollator(max_length=20)
        batch = [
            {
                "heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFTFS",  # 30 chars
                "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQDIS",  # 30 chars
                "heavy_cdr_mask": None,
                "light_cdr_mask": None,
                "heavy_non_templated_mask": None,
                "light_non_templated_mask": None,
            }
        ]

        result = collator(batch)
        # Should be truncated to max_length
        assert result["token_ids"].shape[1] == 20
