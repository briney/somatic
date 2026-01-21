"""Configuration parsing helpers for data loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig


@dataclass
class DatasetConfig:
    """Configuration for a single dataset.

    Attributes:
        path: Path to the dataset file or directory.
        fraction: Sampling fraction for multi-dataset training.
        batch_size: Override batch size for this dataset.
        load_coords: Whether to load 3D coordinates.
        metrics: Metrics configuration for this dataset.
        regions: Region evaluation override for this dataset.
        format: Dataset format, either "sequence" or "structure".
            If None, auto-detected from path.
        chain_id: For structure datasets, specific chain to extract.
        strict: For structure datasets, whether to raise on missing atoms.
        recursive: For structure datasets, whether to search subdirectories.
    """

    path: str
    fraction: float | None = None
    batch_size: int | None = None
    load_coords: bool | None = None
    metrics: dict[str, Any] | None = None
    regions: dict[str, Any] | None = None
    # Structure-specific options
    format: str | None = None  # "sequence" or "structure", auto-detected if None
    chain_id: str | None = None
    strict: bool = False
    recursive: bool = False


def is_single_train_dataset(train_cfg: str | DictConfig | None) -> bool:
    """Check if train config specifies a single dataset.

    Args:
        train_cfg: The train configuration value.

    Returns:
        True if train_cfg is a string path (single dataset), False otherwise.
    """
    return isinstance(train_cfg, str)


def normalize_fractions(fractions: dict[str, float | None]) -> dict[str, float]:
    """Normalize fractions to sum to 1.0.

    Handles cases where:
    - No fractions specified: all datasets get equal weight
    - Some fractions specified: unspecified ones share remaining mass equally
    - All fractions specified: normalize to sum to 1.0

    Args:
        fractions: Dict mapping dataset names to fractions (None for unspecified).

    Returns:
        Dict with all fractions specified and summing to 1.0.

    Examples:
        >>> normalize_fractions({"a": None, "b": None})
        {"a": 0.5, "b": 0.5}

        >>> normalize_fractions({"a": 0.6, "b": 0.4})
        {"a": 0.6, "b": 0.4}

        >>> normalize_fractions({"a": 0.6, "b": None})
        {"a": 0.6, "b": 0.4}

        >>> normalize_fractions({"a": 0.4, "b": None, "c": None})
        {"a": 0.4, "b": 0.3, "c": 0.3}
    """
    if not fractions:
        return {}

    specified = {k: v for k, v in fractions.items() if v is not None}
    unspecified = [k for k, v in fractions.items() if v is None]

    if not specified:
        # No fractions specified - equal weight for all
        n = len(fractions)
        return {k: 1.0 / n for k in fractions}

    specified_sum = sum(specified.values())

    if not unspecified:
        # All specified - normalize to 1.0
        return {k: v / specified_sum for k, v in specified.items()}

    # Some specified, some not - unspecified share remaining mass
    remaining = max(0.0, 1.0 - specified_sum)
    unspecified_fraction = remaining / len(unspecified) if remaining > 0 else 0.0

    result = dict(specified)
    for name in unspecified:
        result[name] = unspecified_fraction

    # Normalize to ensure sum is exactly 1.0
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}

    return result


def parse_train_config(
    train_cfg: str | DictConfig | None,
) -> tuple[dict[str, str], dict[str, float]]:
    """Parse train config into paths and fractions.

    Args:
        train_cfg: Either a string path (single dataset) or a dict of
                   dataset configs (multi-dataset).

    Returns:
        Tuple of (paths dict, fractions dict). For single dataset, the key
        is "train".

    Raises:
        ValueError: If train_cfg is None.

    Examples:
        # Single dataset
        >>> parse_train_config("/path/to/train.parquet")
        ({"train": "/path/to/train.parquet"}, {"train": 1.0})

        # Multi-dataset with fractions
        >>> parse_train_config({
        ...     "dataset_a": {"path": "/a.parquet", "fraction": 0.6},
        ...     "dataset_b": {"path": "/b.parquet", "fraction": 0.4}
        ... })
        ({"dataset_a": "/a.parquet", "dataset_b": "/b.parquet"},
         {"dataset_a": 0.6, "dataset_b": 0.4})

        # Multi-dataset shorthand (just paths)
        >>> parse_train_config({
        ...     "dataset_a": "/a.parquet",
        ...     "dataset_b": "/b.parquet"
        ... })
        ({"dataset_a": "/a.parquet", "dataset_b": "/b.parquet"},
         {"dataset_a": 0.5, "dataset_b": 0.5})
    """
    if train_cfg is None:
        raise ValueError("train config is required")

    # Single dataset case
    if isinstance(train_cfg, str):
        return {"train": train_cfg}, {"train": 1.0}

    # Multi-dataset case
    paths: dict[str, str] = {}
    fractions: dict[str, float | None] = {}

    for name, dataset_cfg in train_cfg.items():
        if isinstance(dataset_cfg, str):
            # Shorthand: just a path
            paths[name] = dataset_cfg
            fractions[name] = None  # Will be filled in by normalize_fractions
        else:
            # Full config with path and optional fraction
            paths[name] = dataset_cfg.get("path") or dataset_cfg.path
            fractions[name] = dataset_cfg.get("fraction")

    # Normalize fractions
    normalized_fractions = normalize_fractions(fractions)

    return paths, normalized_fractions


def is_single_eval_dataset(eval_cfg: str | DictConfig | dict | None) -> bool:
    """Check if eval config specifies a single dataset.

    Args:
        eval_cfg: The eval configuration value.

    Returns:
        True if eval_cfg is a string path (single dataset), False otherwise.
    """
    return isinstance(eval_cfg, str)


def parse_eval_config(
    eval_cfg: str | DictConfig | dict | None,
    global_cfg: DictConfig,
) -> dict[str, DatasetConfig]:
    """Parse eval config into dataset configurations.

    Supports:
    - Single dataset: "/path/to/eval.parquet"
    - Named datasets: {"validation": "/path/to/val.parquet"}
    - Full config: {"validation": {"path": "/path", "batch_size": 32, ...}}

    Dataset format (sequence vs structure) is auto-detected from the path if not
    explicitly specified. CSV, TSV, and Parquet files are treated as sequence
    datasets. Directories containing PDB/mmCIF files are treated as structure
    datasets.

    Args:
        eval_cfg: Either a string path (single dataset) or a dict of
                  dataset configs (multi-dataset).
        global_cfg: Global data configuration for defaults (unused but available
                    for future extensions).

    Returns:
        Dict mapping eval dataset names to DatasetConfig objects. For single
        dataset, the key is "eval".

    Examples:
        # Single dataset
        >>> parse_eval_config("/eval.parquet", cfg)
        {"eval": DatasetConfig(path="/eval.parquet")}

        # Named dataset shorthand
        >>> parse_eval_config({"validation": "/val.parquet"}, cfg)
        {"validation": DatasetConfig(path="/val.parquet")}

        # Full config with format
        >>> parse_eval_config({
        ...     "test": {
        ...         "path": "/test.parquet",
        ...         "batch_size": 64,
        ...         "load_coords": True
        ...     },
        ...     "structures": {
        ...         "path": "/pdb_folder",
        ...         "format": "structure",
        ...         "chain_id": "A"
        ...     }
        ... }, cfg)
        {"test": DatasetConfig(...), "structures": DatasetConfig(...)}
    """
    if not eval_cfg:
        return {}

    # Single dataset case
    if isinstance(eval_cfg, str):
        return {"eval": DatasetConfig(path=eval_cfg)}

    result: dict[str, DatasetConfig] = {}

    for name, cfg in eval_cfg.items():
        if isinstance(cfg, str):
            # Shorthand: just a path
            result[name] = DatasetConfig(path=cfg)
        else:
            # Full config
            result[name] = DatasetConfig(
                path=cfg.get("path") or cfg.path,
                batch_size=cfg.get("batch_size"),
                load_coords=cfg.get("load_coords"),
                metrics=dict(cfg.get("metrics", {})) if cfg.get("metrics") else None,
                regions=dict(cfg.get("regions", {})) if cfg.get("regions") else None,
                format=cfg.get("format"),
                chain_id=cfg.get("chain_id"),
                strict=cfg.get("strict", False),
                recursive=cfg.get("recursive", False),
            )

    return result
