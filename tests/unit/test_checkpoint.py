"""Tests for checkpoint management."""

import pytest
import torch

from somatic.model import SomaticConfig, SomaticForMaskedLM
from somatic.training import CheckpointConfig, CheckpointManager, create_optimizer


@pytest.fixture
def tiny_model():
    """Create a tiny model for checkpoint tests."""
    config = SomaticConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        max_position_embeddings=32,
    )
    return SomaticForMaskedLM(config)


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

        # Checkpoints are HuggingFace directories, not .pt files.
        assert path is not None
        assert path.is_dir()
        assert path.name == "checkpoint_step_100"
        assert (path / "config.json").exists()
        assert (path / "training_state.pt").exists()

    def test_save_with_metrics(self, checkpoint_manager, tmp_path):
        metrics = {"val_loss": 0.5, "val_accuracy": 0.8}
        path = checkpoint_manager.save(step=100, epoch=1, metrics=metrics)

        training_state = torch.load(path / "training_state.pt", weights_only=False)
        assert training_state["metrics"] == metrics

    def test_save_best_checkpoint(self, checkpoint_manager, tmp_path):
        # Save with high loss
        checkpoint_manager.save(step=100, epoch=1, metrics={"val_loss": 1.0})

        # Save with low loss (should become best)
        checkpoint_manager.save(step=200, epoch=2, metrics={"val_loss": 0.5})

        best_dir = checkpoint_manager.save_dir / "best_checkpoint"
        assert best_dir.is_dir()

        best_state = torch.load(best_dir / "training_state.pt", weights_only=False)
        assert best_state["step"] == 200

    def test_keep_last_n(self, checkpoint_manager, tmp_path):
        # Save more checkpoints than keep_last_n
        for i in range(5):
            checkpoint_manager.save(step=(i + 1) * 100, epoch=i + 1)

        # Should only keep last 3 (checkpoint directories)
        checkpoints = [
            p for p in checkpoint_manager.save_dir.glob("checkpoint_step_*") if p.is_dir()
        ]
        assert len(checkpoints) == 3

        # Should have steps 300, 400, 500
        checkpoint_names = {p.name for p in checkpoints}
        assert "checkpoint_step_300" in checkpoint_names
        assert "checkpoint_step_400" in checkpoint_names
        assert "checkpoint_step_500" in checkpoint_names

    def test_load_checkpoint(self, checkpoint_manager, tiny_model, tmp_path):
        # Save checkpoint
        path = checkpoint_manager.save(step=100, epoch=1)

        # Restore training state from the checkpoint directory
        state = checkpoint_manager.load_training_state(str(path))

        assert state["step"] == 100
        assert state["epoch"] == 1

    def test_load_latest(self, checkpoint_manager, tmp_path):
        checkpoint_manager.save(step=100, epoch=1)
        checkpoint_manager.save(step=200, epoch=2)
        checkpoint_manager.save(step=300, epoch=3)

        state = checkpoint_manager.load_training_state()
        assert state["step"] == 300

    def test_load_best(self, checkpoint_manager, tmp_path):
        checkpoint_manager.save(step=100, epoch=1, metrics={"val_loss": 1.0})
        checkpoint_manager.save(step=200, epoch=2, metrics={"val_loss": 0.3})
        checkpoint_manager.save(step=300, epoch=3, metrics={"val_loss": 0.5})

        state = checkpoint_manager.load_training_state(load_best=True)
        assert state["step"] == 200

    def test_load_nonexistent_raises(self, checkpoint_manager, tmp_path):
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_training_state()

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

        best_state = torch.load(
            manager.save_dir / "best_checkpoint" / "training_state.pt", weights_only=False
        )
        assert best_state["step"] == 200
