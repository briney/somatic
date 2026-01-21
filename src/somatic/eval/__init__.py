"""Evaluation harness for computing in-training validation metrics.

This module provides a flexible, extensible system for computing various
metrics during training. It supports:

- Multiple evaluation datasets with independent configurations
- Per-dataset metric selection and parameterization
- Distributed training aggregation
- Metrics computed from logits, embeddings, or attention weights
- Fitting small models (like logistic regression) on representations
- Controlled evaluation masking for reproducible comparisons
- Per-region evaluation for antibody analysis

Example usage:
    from somatic.eval import Evaluator, build_metrics

    # Create evaluator
    evaluator = Evaluator(cfg, model, accelerator)

    # Evaluate on a dataset
    results = evaluator.evaluate(eval_loader, "validation")

    # Or evaluate on all configured datasets
    all_results = evaluator.evaluate_all(eval_loaders)

    # Per-position evaluation
    from somatic.eval import PerPositionEvaluator, AntibodyRegion
    per_pos = PerPositionEvaluator(model)
    results = per_pos.evaluate_by_region(sample)
"""

from .base import Metric, MetricBase
from .evaluator import Evaluator
from .masking import EvalMasker, create_eval_masker
from .per_position import PerPositionEvaluator, RegionMaskingEvaluator
from .region_config import RegionEvalConfig, build_region_eval_config
from .regions import (
    AGGREGATE_GROUP_NAMES,
    AntibodyRegion,
    CDR_REGIONS,
    FWR_REGIONS,
    HEAVY_REGIONS,
    INDIVIDUAL_REGION_NAMES,
    LIGHT_REGIONS,
    aggregate_region_masks,
    extract_region_masks,
)
from .registry import build_metrics, get_metric_class, list_metrics, register_metric

__all__ = [
    # Core classes
    "Metric",
    "MetricBase",
    "Evaluator",
    # Registry functions
    "register_metric",
    "get_metric_class",
    "list_metrics",
    "build_metrics",
    # Masking
    "EvalMasker",
    "create_eval_masker",
    # Region config
    "RegionEvalConfig",
    "build_region_eval_config",
    # Regions
    "AntibodyRegion",
    "CDR_REGIONS",
    "FWR_REGIONS",
    "HEAVY_REGIONS",
    "LIGHT_REGIONS",
    "INDIVIDUAL_REGION_NAMES",
    "AGGREGATE_GROUP_NAMES",
    "extract_region_masks",
    "aggregate_region_masks",
    # Per-position evaluation
    "PerPositionEvaluator",
    "RegionMaskingEvaluator",
]
