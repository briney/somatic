"""Training infrastructure for Somatic."""

from .checkpoint import CheckpointConfig, CheckpointManager
from .flops import FLOPsConfig, FLOPsTracker
from .masking_frequency import MaskingFrequencyConfig, MaskingFrequencyTracker
from .metrics import (
    MLMMetrics,
    MetricAccumulator,
    compute_accuracy,
    compute_masked_cross_entropy,
    compute_mlm_metrics,
    compute_perplexity,
)
from .optimizer import create_optimizer, create_scheduler, get_lr
from .trainer import Trainer, TrainingConfig

__all__ = [
    # Checkpoint
    "CheckpointConfig",
    "CheckpointManager",
    # FLOPs tracking
    "FLOPsConfig",
    "FLOPsTracker",
    # Masking frequency tracking
    "MaskingFrequencyConfig",
    "MaskingFrequencyTracker",
    # Metrics
    "MetricAccumulator",
    "MLMMetrics",
    "compute_masked_cross_entropy",
    "compute_accuracy",
    "compute_perplexity",
    "compute_mlm_metrics",
    # Optimizer
    "create_optimizer",
    "create_scheduler",
    "get_lr",
    # Trainer
    "TrainingConfig",
    "Trainer",
]
