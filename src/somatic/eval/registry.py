"""Metric registration and factory functions."""

from __future__ import annotations

import math
import warnings
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .base import Metric

# Global registry mapping metric names to their classes
METRIC_REGISTRY: dict[str, type[Metric]] = {}


def register_metric(name: str):
    """Decorator to register a metric class in the global registry.

    Args:
        name: Unique identifier for the metric. This is used in config files
            to enable/configure the metric.

    Returns:
        Decorator function that registers the class.

    Example:
        @register_metric("accuracy")
        class AccuracyMetric(MetricBase):
            ...
    """

    def decorator(cls: type[Metric]) -> type[Metric]:
        if name in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered")
        METRIC_REGISTRY[name] = cls
        return cls

    return decorator


def get_metric_class(name: str) -> type[Metric] | None:
    """Get a metric class by name.

    Args:
        name: The registered name of the metric.

    Returns:
        The metric class, or None if not found.
    """
    return METRIC_REGISTRY.get(name)


def list_metrics() -> list[str]:
    """List all registered metric names.

    Returns:
        List of registered metric names.
    """
    return list(METRIC_REGISTRY.keys())


def _get_eval_config(cfg: "DictConfig", eval_name: str | None) -> dict[str, Any]:
    """Get the eval dataset configuration.

    Args:
        cfg: Full configuration object.
        eval_name: Name of the eval dataset, or None for global config.

    Returns:
        Dictionary with eval dataset configuration.
    """
    if eval_name is None:
        return {}

    data_cfg = cfg.get("data", {})
    eval_datasets = data_cfg.get("eval", {})

    if isinstance(eval_datasets, str):
        # Single eval dataset as path string
        return {}

    return dict(eval_datasets.get(eval_name, {}))


def _get_metric_config(
    cfg: "DictConfig",
    metric_name: str,
    eval_name: str | None,
) -> dict[str, Any]:
    """Get merged configuration for a specific metric.

    Configuration is merged in order (later overrides earlier):
    1. Default metric config from eval.metrics.{metric_name}
    2. Per-dataset overrides from data.eval.{eval_name}.metrics.{metric_name}

    Args:
        cfg: Full configuration object.
        metric_name: Name of the metric.
        eval_name: Name of the eval dataset, or None for global only.

    Returns:
        Merged configuration dictionary for the metric.
    """
    result: dict[str, Any] = {"enabled": True}

    # 1. Global defaults from eval.metrics.{metric_name}
    eval_cfg = cfg.get("eval", {})
    global_metrics = eval_cfg.get("metrics", {})
    if metric_name in global_metrics:
        global_metric_cfg = global_metrics[metric_name]
        # Handle both dict and OmegaConf DictConfig
        if isinstance(global_metric_cfg, DictConfig):
            result.update(OmegaConf.to_container(global_metric_cfg))
        elif isinstance(global_metric_cfg, dict):
            result.update(global_metric_cfg)

    # 2. Per-dataset overrides
    if eval_name is not None:
        dataset_cfg = _get_eval_config(cfg, eval_name)
        dataset_metrics = dataset_cfg.get("metrics", {})

        # Check if this metric is in the 'only' whitelist
        only_list = dataset_metrics.get("only")
        if only_list is not None:
            # If 'only' is specified, disable metrics not in the list by default
            if metric_name not in only_list:
                result["enabled"] = False

        # Apply per-metric overrides
        if metric_name in dataset_metrics:
            metric_override = dataset_metrics[metric_name]
            # Handle both dict and OmegaConf DictConfig
            if isinstance(metric_override, DictConfig):
                result.update(OmegaConf.to_container(metric_override))
            elif isinstance(metric_override, dict):
                result.update(metric_override)

    return result


def _get_dataset_has_coords(
    cfg: "DictConfig",
    eval_name: str | None,
    global_has_coords: bool,
) -> bool:
    """Determine if a dataset has coordinates available.

    Args:
        cfg: Full configuration object.
        eval_name: Name of the eval dataset.
        global_has_coords: Global default for coordinate availability.

    Returns:
        Whether coordinates are available for this dataset.
    """
    if eval_name is None:
        return global_has_coords

    dataset_cfg = _get_eval_config(cfg, eval_name)
    return dataset_cfg.get("load_coords", global_has_coords)


def build_metrics(
    cfg: "DictConfig",
    has_coords: bool = False,
    eval_name: str | None = None,
) -> list[Metric]:
    """Build a list of metric instances based on configuration.

    Args:
        cfg: Full configuration object.
        has_coords: Whether coordinate data is available (global default).
        eval_name: Optional eval dataset name for per-dataset metric overrides.

    Returns:
        List of instantiated Metric objects that are enabled and compatible
        with available resources.
    """
    # Import metrics to ensure they're registered
    # This import is deferred to avoid circular imports
    import somatic.eval.metrics  # noqa: F401

    # Resolve per-dataset has_coords override
    dataset_has_coords = _get_dataset_has_coords(cfg, eval_name, has_coords)

    metrics: list[Metric] = []

    for name, cls in METRIC_REGISTRY.items():
        # Get merged config for this metric
        metric_cfg = _get_metric_config(cfg, name, eval_name)

        # Skip if explicitly disabled
        if not metric_cfg.get("enabled", True):
            continue

        # Check resource requirements (using per-dataset has_coords)
        if getattr(cls, "requires_coords", False) and not dataset_has_coords:
            continue

        # Filter out meta-config keys before passing to constructor
        init_kwargs = {
            k: v
            for k, v in metric_cfg.items()
            if k not in ("enabled", "requires_coords", "needs_attentions")
        }

        # Resolve dynamic num_layers for p_at_l metric (null -> 10% of encoder layers)
        if name == "p_at_l" and init_kwargs.get("num_layers") is None:
            model_cfg = cfg.get("model", {})
            n_layers = model_cfg.get("n_layers", 16)
            init_kwargs["num_layers"] = max(1, math.ceil(n_layers * 0.1))

        # Instantiate the metric
        try:
            metric = cls(**init_kwargs)
            metrics.append(metric)
        except Exception as e:
            # Log warning but don't fail - allows graceful degradation
            warnings.warn(f"Failed to instantiate metric '{name}': {e}")

    return metrics
