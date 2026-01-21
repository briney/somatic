"""Tests for data transforms."""

import pytest

from somatic.data.transforms import Compose, RandomChainSwap, SequenceTruncation


class TestRandomChainSwap:
    def test_swap_probability_zero(self):
        transform = RandomChainSwap(p=0.0)
        example = {
            "heavy_chain": "HEAVY",
            "light_chain": "LIGHT",
        }

        result = transform(example)
        assert result["heavy_chain"] == "HEAVY"
        assert result["light_chain"] == "LIGHT"

    def test_swap_probability_one(self):
        transform = RandomChainSwap(p=1.0)
        example = {
            "heavy_chain": "HEAVY",
            "light_chain": "LIGHT",
        }

        result = transform(example)
        assert result["heavy_chain"] == "LIGHT"
        assert result["light_chain"] == "HEAVY"

    def test_swap_with_masks(self):
        transform = RandomChainSwap(p=1.0)
        example = {
            "heavy_chain": "HEAVY",
            "light_chain": "LIGHT",
            "heavy_cdr_mask": [1, 0, 0, 0, 0],
            "light_cdr_mask": [0, 0, 0, 0, 1],
        }

        result = transform(example)
        assert result["heavy_cdr_mask"] == [0, 0, 0, 0, 1]
        assert result["light_cdr_mask"] == [1, 0, 0, 0, 0]

    def test_original_unchanged(self):
        transform = RandomChainSwap(p=1.0)
        example = {
            "heavy_chain": "HEAVY",
            "light_chain": "LIGHT",
        }

        original_heavy = example["heavy_chain"]
        transform(example)

        # Original should be unchanged (we copy)
        assert example["heavy_chain"] == original_heavy


class TestSequenceTruncation:
    def test_no_truncation_needed(self):
        transform = SequenceTruncation(max_length=100)
        example = {
            "heavy_chain": "EVQLVE",
            "light_chain": "DIQMTQ",
        }

        result = transform(example)
        assert result["heavy_chain"] == "EVQLVE"
        assert result["light_chain"] == "DIQMTQ"

    def test_truncation_applied(self):
        transform = SequenceTruncation(max_length=12)  # 10 chars + CLS + EOS
        example = {
            "heavy_chain": "EVQLVESGGGLVQ",  # 13 chars
            "light_chain": "DIQMTQSPSSLSA",  # 13 chars
        }

        result = transform(example)
        total_len = len(result["heavy_chain"]) + len(result["light_chain"])
        assert total_len <= 10

    def test_truncation_with_masks(self):
        transform = SequenceTruncation(max_length=10)
        example = {
            "heavy_chain": "EVQLVE",
            "light_chain": "DIQMTQ",
            "heavy_cdr_mask": [0, 0, 1, 1, 0, 0],
            "light_cdr_mask": [0, 1, 1, 0, 0, 0],
        }

        result = transform(example)
        # Masks should be truncated to match sequence lengths
        if result["heavy_cdr_mask"] is not None:
            assert len(result["heavy_cdr_mask"]) == len(result["heavy_chain"])
        if result["light_cdr_mask"] is not None:
            assert len(result["light_cdr_mask"]) == len(result["light_chain"])


class TestCompose:
    def test_compose_multiple(self):
        transform = Compose([
            RandomChainSwap(p=1.0),
            SequenceTruncation(max_length=100),
        ])

        example = {
            "heavy_chain": "HEAVY",
            "light_chain": "LIGHT",
        }

        result = transform(example)
        # After swap
        assert result["heavy_chain"] == "LIGHT"
        assert result["light_chain"] == "HEAVY"

    def test_compose_empty(self):
        transform = Compose([])
        example = {"heavy_chain": "HEAVY", "light_chain": "LIGHT"}

        result = transform(example)
        assert result == example
