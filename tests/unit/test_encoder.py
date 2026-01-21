"""Tests for SomaticEncoder log_likelihood and perplexity methods."""

import pytest
import torch

from somatic.encoding import SomaticEncoder
from somatic.model import SomaticConfig, SomaticModel


# Sample antibody sequences for testing
HEAVY_CHAIN = "EVQLVQSGAEVKKPGESLKISCKGSGYSFT"
LIGHT_CHAIN = "DIQMTQSPSSLSASVGDRVTITC"


@pytest.fixture
def encoder(small_model: SomaticModel) -> SomaticEncoder:
    """Create an encoder from the small model fixture."""
    return SomaticEncoder(small_model, device="cpu")


class TestLogLikelihood:
    """Tests for the log_likelihood method."""

    def test_returns_expected_keys(self, encoder: SomaticEncoder):
        """log_likelihood should return dict with expected keys."""
        result = encoder.log_likelihood(HEAVY_CHAIN, LIGHT_CHAIN)

        assert "log_likelihood" in result
        assert "heavy_log_likelihood" in result
        assert "light_log_likelihood" in result

    def test_values_are_negative(self, encoder: SomaticEncoder):
        """Log-likelihood values should be negative (log probs <= 0)."""
        result = encoder.log_likelihood(HEAVY_CHAIN, LIGHT_CHAIN)

        assert result["log_likelihood"].item() <= 0
        assert result["heavy_log_likelihood"].item() <= 0
        assert result["light_log_likelihood"].item() <= 0

    def test_values_are_scalars(self, encoder: SomaticEncoder):
        """Log-likelihood values should be scalar tensors."""
        result = encoder.log_likelihood(HEAVY_CHAIN, LIGHT_CHAIN)

        assert result["log_likelihood"].dim() == 0
        assert result["heavy_log_likelihood"].dim() == 0
        assert result["light_log_likelihood"].dim() == 0

    def test_heavy_plus_light_equals_total(self, encoder: SomaticEncoder):
        """Heavy + light log-likelihood should equal total."""
        result = encoder.log_likelihood(HEAVY_CHAIN, LIGHT_CHAIN)

        total = result["log_likelihood"].item()
        heavy = result["heavy_log_likelihood"].item()
        light = result["light_log_likelihood"].item()

        assert abs(total - (heavy + light)) < 1e-5

    def test_different_sequences_give_different_values(
        self, encoder: SomaticEncoder
    ):
        """Different sequences should give different log-likelihood values."""
        result1 = encoder.log_likelihood(HEAVY_CHAIN, LIGHT_CHAIN)
        result2 = encoder.log_likelihood("AAAAAAAAAA", "CCCCCCCCCC")

        assert result1["log_likelihood"].item() != result2["log_likelihood"].item()


class TestPerplexity:
    """Tests for the perplexity method."""

    def test_returns_expected_keys(self, encoder: SomaticEncoder):
        """perplexity should return dict with expected keys."""
        result = encoder.perplexity(HEAVY_CHAIN, LIGHT_CHAIN)

        assert "perplexity" in result
        assert "heavy_perplexity" in result
        assert "light_perplexity" in result

    def test_values_are_at_least_one(self, encoder: SomaticEncoder):
        """Perplexity values should be >= 1.0."""
        result = encoder.perplexity(HEAVY_CHAIN, LIGHT_CHAIN)

        assert result["perplexity"].item() >= 1.0
        assert result["heavy_perplexity"].item() >= 1.0
        assert result["light_perplexity"].item() >= 1.0

    def test_values_are_scalars(self, encoder: SomaticEncoder):
        """Perplexity values should be scalar tensors."""
        result = encoder.perplexity(HEAVY_CHAIN, LIGHT_CHAIN)

        assert result["perplexity"].dim() == 0
        assert result["heavy_perplexity"].dim() == 0
        assert result["light_perplexity"].dim() == 0

    def test_perplexity_consistent_with_log_likelihood(
        self, encoder: SomaticEncoder
    ):
        """Perplexity should be exp(-log_likelihood / num_tokens)."""
        ll_result = encoder.log_likelihood(HEAVY_CHAIN, LIGHT_CHAIN)
        ppl_result = encoder.perplexity(HEAVY_CHAIN, LIGHT_CHAIN)

        heavy_len = len(HEAVY_CHAIN)
        light_len = len(LIGHT_CHAIN)
        total_len = heavy_len + light_len

        expected_heavy_ppl = torch.exp(
            -ll_result["heavy_log_likelihood"] / heavy_len
        ).item()
        expected_light_ppl = torch.exp(
            -ll_result["light_log_likelihood"] / light_len
        ).item()
        expected_total_ppl = torch.exp(
            -ll_result["log_likelihood"] / total_len
        ).item()

        assert abs(ppl_result["heavy_perplexity"].item() - expected_heavy_ppl) < 1e-5
        assert abs(ppl_result["light_perplexity"].item() - expected_light_ppl) < 1e-5
        assert abs(ppl_result["perplexity"].item() - expected_total_ppl) < 1e-5

    def test_different_sequences_give_different_values(
        self, encoder: SomaticEncoder
    ):
        """Different sequences should give different perplexity values."""
        result1 = encoder.perplexity(HEAVY_CHAIN, LIGHT_CHAIN)
        result2 = encoder.perplexity("AAAAAAAAAA", "CCCCCCCCCC")

        assert result1["perplexity"].item() != result2["perplexity"].item()
