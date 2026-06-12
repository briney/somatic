"""Integration tests for the masking pipeline."""

import pytest
import torch

from somatic.masking import InformationWeightedMasker, UniformMasker
from somatic.model import SomaticConfig, SomaticForMaskedLM
from somatic.model.tokenization_somatic import tokenizer


@pytest.fixture
def model():
    """Create a small masked-LM model for testing."""
    config = SomaticConfig(
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    return SomaticForMaskedLM(config)


@pytest.fixture
def sample_batch():
    """Create a sample batch with CDR masks."""
    heavy = "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH"
    light = "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLA"

    heavy_ids = tokenizer.encode(heavy, add_special_tokens=False)
    light_ids = tokenizer.encode(light, add_special_tokens=False)

    tokens = [tokenizer.cls_token_id] + heavy_ids + light_ids + [tokenizer.eos_token_id]
    chains = [0] * (1 + len(heavy_ids)) + [1] * (len(light_ids) + 1)

    # Create CDR mask (positions 10-15 and 25-30 are CDRs)
    cdr_mask = [0] * len(tokens)
    for i in range(10, 16):
        if i < len(tokens):
            cdr_mask[i] = 1
    for i in range(25, 31):
        if i < len(tokens):
            cdr_mask[i] = 1

    # Special tokens mask
    special_mask = [True] + [False] * (len(tokens) - 2) + [True]

    return {
        "input_ids": torch.tensor([tokens, tokens]),
        "token_type_ids": torch.tensor([chains, chains]),
        "attention_mask": torch.ones(2, len(tokens)),
        "cdr_mask": torch.tensor([cdr_mask, cdr_mask]),
        "special_tokens_mask": torch.tensor([special_mask, special_mask]),
    }


class TestMLMTrainingStep:
    def test_uniform_masking_forward_loss(self, model, sample_batch):
        """Test full training step with uniform masking."""
        masker = UniformMasker(mask_rate=0.15)

        # Apply masking
        masked_ids, labels = masker.apply_mask(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            special_tokens_mask=sample_batch["special_tokens_mask"],
        )

        # Forward pass with labels -> model computes loss internally
        outputs = model(
            input_ids=masked_ids,
            token_type_ids=sample_batch["token_type_ids"],
            attention_mask=sample_batch["attention_mask"],
            labels=labels,
        )

        loss = outputs.loss

        assert loss.ndim == 0
        assert loss > 0
        assert not torch.isnan(loss)

    def test_information_weighted_masking_forward_loss(self, model, sample_batch):
        """Test full training step with information-weighted masking."""
        torch.manual_seed(42)

        masker = InformationWeightedMasker(
            mask_rate=0.15,
            cdr_weight_multiplier=2.0,
            nongermline_weight_multiplier=1.0,
        )

        # Apply masking with CDR weighting
        masked_ids, labels = masker.apply_mask(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            cdr_mask=sample_batch["cdr_mask"],
            special_tokens_mask=sample_batch["special_tokens_mask"],
        )

        # Forward pass with labels -> model computes loss internally
        outputs = model(
            input_ids=masked_ids,
            token_type_ids=sample_batch["token_type_ids"],
            attention_mask=sample_batch["attention_mask"],
            labels=labels,
        )

        loss = outputs.loss

        assert loss.ndim == 0
        assert loss > 0

    def test_training_step_gradient_update(self, model, sample_batch):
        """Test that a training step updates parameters."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        masker = UniformMasker(mask_rate=0.15)

        # Get initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        # Training step
        masked_ids, labels = masker.apply_mask(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            special_tokens_mask=sample_batch["special_tokens_mask"],
        )

        outputs = model(
            input_ids=masked_ids,
            token_type_ids=sample_batch["token_type_ids"],
            attention_mask=sample_batch["attention_mask"],
            labels=labels,
        )

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that parameters changed
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break

        assert params_changed

    def test_sampled_masking_forward_loss(self, model, sample_batch):
        """Test training step with Gumbel-top-k sampled masking produces valid loss."""
        torch.manual_seed(42)

        masker = InformationWeightedMasker(
            mask_rate=0.15,
            cdr_weight_multiplier=2.0,
            nongermline_weight_multiplier=1.0,
            selection_method="sampled",
        )

        masked_ids, labels = masker.apply_mask(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            cdr_mask=sample_batch["cdr_mask"],
            special_tokens_mask=sample_batch["special_tokens_mask"],
        )

        outputs = model(
            input_ids=masked_ids,
            token_type_ids=sample_batch["token_type_ids"],
            attention_mask=sample_batch["attention_mask"],
            labels=labels,
        )

        loss = outputs.loss

        assert loss.ndim == 0
        assert loss > 0
        assert not torch.isnan(loss)

    def test_ranked_masking_forward_loss(self, model, sample_batch):
        """Test training step with ranked masking produces valid loss."""
        torch.manual_seed(42)

        masker = InformationWeightedMasker(
            mask_rate=0.15,
            cdr_weight_multiplier=2.0,
            nongermline_weight_multiplier=1.0,
            selection_method="ranked",
        )

        masked_ids, labels = masker.apply_mask(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            cdr_mask=sample_batch["cdr_mask"],
            special_tokens_mask=sample_batch["special_tokens_mask"],
        )

        outputs = model(
            input_ids=masked_ids,
            token_type_ids=sample_batch["token_type_ids"],
            attention_mask=sample_batch["attention_mask"],
            labels=labels,
        )

        loss = outputs.loss

        assert loss.ndim == 0
        assert loss > 0
        assert not torch.isnan(loss)

    @pytest.mark.parametrize("selection_method", ["sampled", "ranked"])
    def test_selection_method_gradient_update(self, model, sample_batch, selection_method):
        """Test that both selection methods update model parameters correctly."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        masker = InformationWeightedMasker(
            mask_rate=0.15,
            cdr_weight_multiplier=2.0,
            nongermline_weight_multiplier=1.0,
            selection_method=selection_method,
        )

        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        masked_ids, labels = masker.apply_mask(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            cdr_mask=sample_batch["cdr_mask"],
            special_tokens_mask=sample_batch["special_tokens_mask"],
        )

        outputs = model(
            input_ids=masked_ids,
            token_type_ids=sample_batch["token_type_ids"],
            attention_mask=sample_batch["attention_mask"],
            labels=labels,
        )

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break

        assert params_changed, f"Parameters should change with {selection_method} selection"


class TestPredictMasked:
    def test_predict_masked_fills_masks(self, model, sample_batch):
        """Test that predict_masked fills in mask tokens."""
        model.eval()

        # Create input with some mask tokens
        input_ids = sample_batch["input_ids"][0:1].clone()
        token_type_ids = sample_batch["token_type_ids"][0:1]

        # Mask positions 5-10
        original_tokens = input_ids.clone()
        input_ids[0, 5:10] = tokenizer.mask_token_id

        with torch.no_grad():
            predicted = model.predict_masked(
                input_ids,
                token_type_ids=token_type_ids,
            )

        # Check that at least some masked positions are filled
        # (with an untrained model, some might still be mask tokens)
        mask_positions = predicted[0, 5:10]
        assert (mask_positions != tokenizer.mask_token_id).any(), (
            "At least some masks should be filled"
        )

        # Check that unmasked positions are preserved
        assert torch.equal(predicted[0, :5], original_tokens[0, :5])
        assert torch.equal(predicted[0, 10:], original_tokens[0, 10:])

    def test_predict_masked_with_temperature(self, model, sample_batch):
        """Test predict_masked with temperature parameter."""
        model.eval()

        input_ids = sample_batch["input_ids"][0:1].clone()
        token_type_ids = sample_batch["token_type_ids"][0:1]
        input_ids[0, 5:10] = tokenizer.mask_token_id

        with torch.no_grad():
            # Low temperature should be more deterministic
            predicted_low_temp = model.predict_masked(
                input_ids.clone(),
                token_type_ids=token_type_ids,
                temperature=0.1,
            )
            # High temperature should be more random
            predicted_high_temp = model.predict_masked(
                input_ids.clone(),
                token_type_ids=token_type_ids,
                temperature=2.0,
            )

        # Both should produce valid outputs
        assert (predicted_low_temp >= 0).all()
        assert (predicted_high_temp >= 0).all()
        assert (predicted_low_temp < 32).all()
        assert (predicted_high_temp < 32).all()


class TestMaskRateVariations:
    @pytest.mark.parametrize("mask_rate", [0.05, 0.15, 0.30, 0.50])
    def test_different_mask_rates(self, model, sample_batch, mask_rate):
        """Test that different mask rates work correctly."""
        masker = UniformMasker(mask_rate=mask_rate)

        masked_ids, labels = masker.apply_mask(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            special_tokens_mask=sample_batch["special_tokens_mask"],
        )

        outputs = model(
            input_ids=masked_ids,
            token_type_ids=sample_batch["token_type_ids"],
            attention_mask=sample_batch["attention_mask"],
            labels=labels,
        )

        loss = outputs.loss

        assert not torch.isnan(loss), f"NaN loss for mask_rate: {mask_rate}"
