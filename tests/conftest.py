"""Pytest fixtures for Somatic tests."""

import pytest
import torch

from somatic.model import SomaticConfig, SomaticModel


@pytest.fixture
def small_config() -> SomaticConfig:
    return SomaticConfig(
        vocab_size=32,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=64,
        dropout=0.0,
        attention_dropout=0.0,
        embedding_dropout=0.0,
    )


@pytest.fixture
def small_model(small_config: SomaticConfig) -> SomaticModel:
    return SomaticModel(small_config)


@pytest.fixture
def sample_batch() -> dict[str, torch.Tensor]:
    """Create a simple sample batch for testing."""
    batch_size, seq_len = 2, 32
    token_ids = torch.randint(4, 28, (batch_size, seq_len))  # Amino acid tokens
    # Add CLS at start, EOS at end
    token_ids[:, 0] = 0  # CLS
    token_ids[:, -1] = 2  # EOS

    # Chain IDs: first half is chain 0, second half is chain 1
    chain_ids = torch.cat(
        [torch.zeros(batch_size, seq_len // 2), torch.ones(batch_size, seq_len // 2)],
        dim=1,
    ).long()

    # Attention mask: all ones (no padding)
    attention_mask = torch.ones(batch_size, seq_len)

    return {
        "token_ids": token_ids,
        "chain_ids": chain_ids,
        "attention_mask": attention_mask,
    }
