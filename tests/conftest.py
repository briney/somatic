"""Pytest fixtures for Somatic tests."""

import pytest
import torch

from somatic.model import SomaticConfig, SomaticForMaskedLM, SomaticModel


@pytest.fixture
def small_config() -> SomaticConfig:
    return SomaticConfig(
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


@pytest.fixture
def small_model(small_config: SomaticConfig) -> SomaticForMaskedLM:
    """Tiny masked-LM model (has the lm_head; logits/predict available)."""
    return SomaticForMaskedLM(small_config)


@pytest.fixture
def small_base_model(small_config: SomaticConfig) -> SomaticModel:
    """Tiny base encoder (no task head; returns ``last_hidden_state``)."""
    return SomaticModel(small_config)


@pytest.fixture
def sample_batch() -> dict[str, torch.Tensor]:
    """Create a simple sample batch for testing."""
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(4, 28, (batch_size, seq_len))  # Amino acid tokens
    # Add CLS at start, EOS at end
    input_ids[:, 0] = 0  # CLS
    input_ids[:, -1] = 2  # EOS

    # token_type_ids: first half is chain 0, second half is chain 1
    token_type_ids = torch.cat(
        [torch.zeros(batch_size, seq_len // 2), torch.ones(batch_size, seq_len // 2)],
        dim=1,
    ).long()

    # Attention mask: all ones (no padding)
    attention_mask = torch.ones(batch_size, seq_len)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }
