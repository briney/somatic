"""Tests for checkpoint management."""

import pytest
import torch

from somatic.model import SomaticConfig, SomaticModel
from somatic.training import CheckpointConfig, CheckpointManager, create_optimizer


@pytest.fixture
def tiny_model():
    """Create a tiny model for checkpoint tests."""
    config = SomaticConfig(
        vocab_size=32,
        d_model=32,
        n_layers=1,
        n_heads=1,
        max_seq_len=32,
    )
    return SomaticModel(config)


@pytest.fixture
def checkpoint_manager(tiny_model, tmp_path):
    """Create a checkpoint manager with temporary directory."""
    config = CheckpointConfig(
        save_dir=str(tmp_path / "checkpoints"),
        checkpoint_steps=100,
        keep_last_n=3,
        save_best=True,
        best_metric="val_loss",
        best_mode="min",
    )
    optimizer = create_optimizer(tiny_model)
    return CheckpointManager(config, tiny_model, optimizer)


class TestCheckpointConfig:
    def test_default_values(self):
        config = CheckpointConfig()
        assert config.save_dir == "checkpoints"
        assert config.checkpoint_steps == 1000
        assert config.keep_last_n == 5
        assert config.save_best is True
        assert config.best_metric == "val_loss"
        assert config.best_mode == "min"


class TestCheckpointManager:
    def test_save_checkpoint(self, checkpoint_manager, tmp_path):
        path = checkpoint_manager.save(step=100, epoch=1)

        assert path is not None
        assert path.exists()
        assert "checkpoint_step_100.pt" in str(path)

    def test_save_with_metrics(self, checkpoint_manager, tmp_path):
        metrics = {"val_loss": 0.5, "val_accuracy": 0.8}
        path = checkpoint_manager.save(step=100, epoch=1, metrics=metrics)

        checkpoint = torch.load(path, weights_only=False)
        assert checkpoint["metrics"] == metrics

    def test_save_best_checkpoint(self, checkpoint_manager, tmp_path):
        # Save with high loss
        checkpoint_manager.save(step=100, epoch=1, metrics={"val_loss": 1.0})

        # Save with low loss (should become best)
        checkpoint_manager.save(step=200, epoch=2, metrics={"val_loss": 0.5})

        best_path = checkpoint_manager.save_dir / "best_checkpoint.pt"
        assert best_path.exists()

        best_checkpoint = torch.load(best_path, weights_only=False)
        assert best_checkpoint["step"] == 200

    def test_keep_last_n(self, checkpoint_manager, tmp_path):
        # Save more checkpoints than keep_last_n
        for i in range(5):
            checkpoint_manager.save(step=(i + 1) * 100, epoch=i + 1)

        # Should only keep last 3
        checkpoints = list(checkpoint_manager.save_dir.glob("checkpoint_step_*.pt"))
        assert len(checkpoints) == 3

        # Should have steps 300, 400, 500
        checkpoint_names = {p.name for p in checkpoints}
        assert "checkpoint_step_300.pt" in checkpoint_names
        assert "checkpoint_step_400.pt" in checkpoint_names
        assert "checkpoint_step_500.pt" in checkpoint_names

    def test_load_checkpoint(self, checkpoint_manager, tiny_model, tmp_path):
        # Save checkpoint
        path = checkpoint_manager.save(step=100, epoch=1)

        # Load checkpoint
        state = checkpoint_manager.load(str(path))

        assert state["step"] == 100
        assert state["epoch"] == 1

    def test_load_latest(self, checkpoint_manager, tmp_path):
        checkpoint_manager.save(step=100, epoch=1)
        checkpoint_manager.save(step=200, epoch=2)
        checkpoint_manager.save(step=300, epoch=3)

        state = checkpoint_manager.load()
        assert state["step"] == 300

    def test_load_best(self, checkpoint_manager, tmp_path):
        checkpoint_manager.save(step=100, epoch=1, metrics={"val_loss": 1.0})
        checkpoint_manager.save(step=200, epoch=2, metrics={"val_loss": 0.3})
        checkpoint_manager.save(step=300, epoch=3, metrics={"val_loss": 0.5})

        state = checkpoint_manager.load(load_best=True)
        assert state["step"] == 200

    def test_load_nonexistent_raises(self, checkpoint_manager, tmp_path):
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load()

    def test_should_save(self, checkpoint_manager):
        # Default is every 100 steps
        config = CheckpointConfig(checkpoint_steps=100)
        checkpoint_manager.config = config

        assert checkpoint_manager.should_save(0) is False
        assert checkpoint_manager.should_save(50) is False
        assert checkpoint_manager.should_save(100) is True
        assert checkpoint_manager.should_save(200) is True
        assert checkpoint_manager.should_save(150) is False

    def test_best_mode_max(self, tiny_model, tmp_path):
        """Test best checkpoint tracking with max mode."""
        config = CheckpointConfig(
            save_dir=str(tmp_path / "checkpoints"),
            save_best=True,
            best_metric="val_accuracy",
            best_mode="max",
        )
        optimizer = create_optimizer(tiny_model)
        manager = CheckpointManager(config, tiny_model, optimizer)

        # Save with low accuracy
        manager.save(step=100, epoch=1, metrics={"val_accuracy": 0.5})

        # Save with high accuracy (should become best)
        manager.save(step=200, epoch=2, metrics={"val_accuracy": 0.9})

        # Save with medium accuracy (should not become best)
        manager.save(step=300, epoch=3, metrics={"val_accuracy": 0.7})

        best_checkpoint = torch.load(
            manager.save_dir / "best_checkpoint.pt", weights_only=False
        )
        assert best_checkpoint["step"] == 200

    def test_extra_state(self, checkpoint_manager, tmp_path):
        extra = {"custom_data": [1, 2, 3], "epoch_losses": [0.5, 0.4, 0.3]}
        checkpoint_manager.save(step=100, epoch=1, extra_state=extra)

        state = checkpoint_manager.load()
        assert state["extra_state"]["custom_data"] == [1, 2, 3]
        assert state["extra_state"]["epoch_losses"] == [0.5, 0.4, 0.3]
