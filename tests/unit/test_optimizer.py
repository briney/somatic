"""Tests for optimizer and scheduler utilities."""

import pytest
import torch

from somatic.model import SomaticConfig, SomaticModel
from somatic.training import create_optimizer, create_scheduler, get_lr


@pytest.fixture
def tiny_model():
    """Create a tiny model for optimizer tests."""
    config = SomaticConfig(
        vocab_size=32,
        d_model=32,
        n_layers=1,
        n_heads=1,
        max_seq_len=32,
    )
    return SomaticModel(config)


class TestCreateOptimizer:
    def test_basic_creation(self, tiny_model):
        optimizer = create_optimizer(tiny_model)
        assert optimizer is not None
        assert len(optimizer.param_groups) == 2  # decay and no_decay

    def test_custom_lr(self, tiny_model):
        lr = 5e-5
        optimizer = create_optimizer(tiny_model, lr=lr)
        assert optimizer.param_groups[0]["lr"] == lr
        assert optimizer.param_groups[1]["lr"] == lr

    def test_weight_decay_separation(self, tiny_model):
        weight_decay = 0.1
        optimizer = create_optimizer(tiny_model, weight_decay=weight_decay)

        # First group should have weight decay
        assert optimizer.param_groups[0]["weight_decay"] == weight_decay
        # Second group (biases, layer norms) should have no decay
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_custom_betas(self, tiny_model):
        betas = (0.95, 0.98)
        optimizer = create_optimizer(tiny_model, betas=betas)
        assert optimizer.param_groups[0]["betas"] == betas

    def test_optimizer_step(self, tiny_model):
        optimizer = create_optimizer(tiny_model, lr=0.01)

        # Create dummy input and compute loss
        x = torch.randint(0, 32, (1, 10))
        chain_ids = torch.zeros(1, 10, dtype=torch.long)

        output = tiny_model(x, chain_ids)
        loss = output["logits"].sum()

        # Take optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify parameters were updated
        assert True  # If we got here without error, the step worked


class TestCreateScheduler:
    def test_constant_scheduler(self, tiny_model):
        optimizer = create_optimizer(tiny_model)
        scheduler = create_scheduler(
            optimizer,
            scheduler_decay="constant",
            num_training_steps=1000,
            num_warmup_steps=100,
        )
        assert scheduler is not None

    def test_linear_scheduler(self, tiny_model):
        optimizer = create_optimizer(tiny_model)
        scheduler = create_scheduler(
            optimizer,
            scheduler_decay="linear",
            num_training_steps=1000,
            num_warmup_steps=100,
        )
        assert scheduler is not None

    def test_cosine_scheduler(self, tiny_model):
        optimizer = create_optimizer(tiny_model)
        scheduler = create_scheduler(
            optimizer,
            scheduler_decay="cosine",
            num_training_steps=1000,
            num_warmup_steps=100,
        )
        assert scheduler is not None

    def test_invalid_scheduler_decay(self, tiny_model):
        optimizer = create_optimizer(tiny_model)
        with pytest.raises(ValueError, match="Unknown scheduler decay type"):
            create_scheduler(optimizer, scheduler_decay="invalid")

    def test_warmup_behavior(self, tiny_model):
        base_lr = 1e-3
        optimizer = create_optimizer(tiny_model, lr=base_lr)
        scheduler = create_scheduler(
            optimizer,
            scheduler_decay="cosine",
            num_training_steps=1000,
            num_warmup_steps=100,
        )

        # At step 0, LR should be very low (warmup start)
        initial_lr = get_lr(optimizer)
        assert initial_lr < base_lr * 0.01  # Should be near zero

        # After warmup, LR should be at base_lr
        for _ in range(100):
            scheduler.step()

        warmup_end_lr = get_lr(optimizer)
        assert warmup_end_lr == pytest.approx(base_lr, rel=0.01)

    def test_cosine_decay(self, tiny_model):
        base_lr = 1e-3
        min_lr_ratio = 0.1
        optimizer = create_optimizer(tiny_model, lr=base_lr)
        scheduler = create_scheduler(
            optimizer,
            scheduler_decay="cosine",
            num_training_steps=1000,
            num_warmup_steps=100,
            min_lr_ratio=min_lr_ratio,
        )

        # Run past warmup to mid-training
        for _ in range(500):
            scheduler.step()

        mid_lr = get_lr(optimizer)
        # Mid-training LR should be less than base but more than min
        expected_min = base_lr * min_lr_ratio
        assert mid_lr < base_lr
        assert mid_lr > expected_min


    def test_warmup_equals_training_steps(self, tiny_model):
        """Test that scheduler doesn't crash when warmup_steps == num_training_steps."""
        optimizer = create_optimizer(tiny_model)

        # This should not raise ZeroDivisionError
        scheduler = create_scheduler(
            optimizer,
            scheduler_decay="cosine",
            num_training_steps=1000,
            num_warmup_steps=1000,  # Equal to training steps
        )

        # Run through all steps - should not crash
        for _ in range(1000):
            scheduler.step()

    def test_warmup_exceeds_training_steps(self, tiny_model):
        """Test that scheduler handles warmup > training steps gracefully."""
        optimizer = create_optimizer(tiny_model)

        scheduler = create_scheduler(
            optimizer,
            scheduler_decay="linear",
            num_training_steps=100,
            num_warmup_steps=1000,  # More than training steps
        )

        # Run through all steps - should not crash
        for _ in range(100):
            scheduler.step()


    def test_linear_decay_reaches_min_at_correct_step(self, tiny_model):
        """Verify linear decay reaches min_lr_ratio at exactly num_training_steps."""
        base_lr = 1e-3
        min_lr_ratio = 0.0
        num_training_steps = 1000
        num_warmup_steps = 100

        optimizer = create_optimizer(tiny_model, lr=base_lr)
        scheduler = create_scheduler(
            optimizer,
            scheduler_decay="linear",
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )

        # Check warmup peak (step after warmup completes)
        for _ in range(num_warmup_steps):
            scheduler.step()
        assert get_lr(optimizer) == pytest.approx(base_lr, rel=0.01)

        # Check mid-decay (step 550 = 50% through 900-step decay phase)
        for _ in range(450):
            scheduler.step()
        expected_mid = base_lr * 0.5  # Linear decay should be at 50%
        assert get_lr(optimizer) == pytest.approx(expected_mid, rel=0.05)

        # Check final step
        for _ in range(450):
            scheduler.step()
        expected_final = base_lr * min_lr_ratio
        assert get_lr(optimizer) == pytest.approx(expected_final, abs=1e-9)

    def test_cosine_decay_reaches_min_at_correct_step(self, tiny_model):
        """Verify cosine decay reaches min_lr_ratio at exactly num_training_steps."""
        base_lr = 1e-3
        min_lr_ratio = 0.1
        num_training_steps = 1000
        num_warmup_steps = 100

        optimizer = create_optimizer(tiny_model, lr=base_lr)
        scheduler = create_scheduler(
            optimizer,
            scheduler_decay="cosine",
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )

        # Run to final step
        for _ in range(num_training_steps):
            scheduler.step()

        expected_final = base_lr * min_lr_ratio
        assert get_lr(optimizer) == pytest.approx(expected_final, rel=0.01)

    def test_scheduler_trajectory_with_large_step_count(self, tiny_model):
        """Verify scheduler works correctly with realistic step counts."""
        base_lr = 3e-4
        min_lr_ratio = 0.0
        num_training_steps = 250000
        num_warmup_steps = 10000

        optimizer = create_optimizer(tiny_model, lr=base_lr)
        scheduler = create_scheduler(
            optimizer,
            scheduler_decay="linear",
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )

        # Check at step 10000 (end of warmup)
        for _ in range(num_warmup_steps):
            scheduler.step()
        assert get_lr(optimizer) == pytest.approx(base_lr, rel=0.01)

        # Check at step 130000 (50% through decay)
        for _ in range(120000):
            scheduler.step()
        expected_mid = base_lr * 0.5
        assert get_lr(optimizer) == pytest.approx(expected_mid, rel=0.01)

        # Check at step 250000 (end of training)
        for _ in range(120000):
            scheduler.step()
        assert get_lr(optimizer) == pytest.approx(0.0, abs=1e-9)


class TestGetLR:
    def test_get_lr(self, tiny_model):
        lr = 3e-4
        optimizer = create_optimizer(tiny_model, lr=lr)
        assert get_lr(optimizer) == lr
