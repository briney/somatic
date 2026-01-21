"""DataLoader factory with support for weighted sampling."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig
from torch.utils.data import DataLoader, WeightedRandomSampler

from .collator import AntibodyCollator
from .config import is_single_train_dataset, parse_eval_config, parse_train_config
from .dataset import (
    AntibodyDataset,
    MultiDataset,
    StructureDataset,
    detect_dataset_format,
)


def create_dataloader(
    data_path: str | Path,
    batch_size: int,
    max_length: int = 320,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    pad_to_max: bool = False,
    load_coords: bool = False,
    heavy_col: str = "heavy_chain",
    light_col: str = "light_chain",
    heavy_cdr_col: str = "heavy_cdr_mask",
    light_cdr_col: str = "light_cdr_mask",
    heavy_nongermline_col: str = "heavy_non_templated_mask",
    light_nongermline_col: str = "light_non_templated_mask",
    heavy_coords_col: str = "heavy_coords",
    light_coords_col: str = "light_coords",
) -> DataLoader:
    """
    Create a DataLoader for antibody sequence data.

    Args:
        data_path: Path to CSV or Parquet file.
        batch_size: Batch size.
        max_length: Maximum sequence length.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for faster GPU transfer.
        drop_last: Whether to drop the last incomplete batch.
        pad_to_max: Whether to always pad to max_length.
        load_coords: Whether to load 3D coordinates.
        heavy_col: Column name for heavy chain sequences.
        light_col: Column name for light chain sequences.
        heavy_cdr_col: Column name for heavy chain CDR mask.
        light_cdr_col: Column name for light chain CDR mask.
        heavy_nongermline_col: Column name for heavy chain non-templated mask.
        light_nongermline_col: Column name for light chain non-templated mask.
        heavy_coords_col: Column name for heavy chain coordinates.
        light_coords_col: Column name for light chain coordinates.

    Returns:
        DataLoader instance.
    """
    dataset = AntibodyDataset(
        data_path,
        max_length=max_length,
        heavy_col=heavy_col,
        light_col=light_col,
        heavy_cdr_col=heavy_cdr_col,
        light_cdr_col=light_cdr_col,
        heavy_nongermline_col=heavy_nongermline_col,
        light_nongermline_col=light_nongermline_col,
        load_coords=load_coords,
        heavy_coords_col=heavy_coords_col,
        light_coords_col=light_coords_col,
    )
    collator = AntibodyCollator(max_length=max_length, pad_to_max=pad_to_max)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator,
    )


def create_multi_dataloader(
    data_paths: dict[str, str | Path],
    weights: dict[str, float] | None,
    batch_size: int,
    max_length: int = 320,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    pad_to_max: bool = False,
    load_coords: bool = False,
    heavy_col: str = "heavy_chain",
    light_col: str = "light_chain",
    heavy_cdr_col: str = "heavy_cdr_mask",
    light_cdr_col: str = "light_cdr_mask",
    heavy_nongermline_col: str = "heavy_non_templated_mask",
    light_nongermline_col: str = "light_non_templated_mask",
    heavy_coords_col: str = "heavy_coords",
    light_coords_col: str = "light_coords",
) -> DataLoader:
    """
    Create a DataLoader for multiple datasets with weighted sampling.

    Args:
        data_paths: Dictionary mapping dataset names to file paths.
        weights: Optional dictionary mapping dataset names to sampling weights.
        batch_size: Batch size.
        max_length: Maximum sequence length.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for faster GPU transfer.
        drop_last: Whether to drop the last incomplete batch.
        pad_to_max: Whether to always pad to max_length.
        load_coords: Whether to load 3D coordinates.
        heavy_col: Column name for heavy chain sequences.
        light_col: Column name for light chain sequences.
        heavy_cdr_col: Column name for heavy chain CDR mask.
        light_cdr_col: Column name for light chain CDR mask.
        heavy_nongermline_col: Column name for heavy chain non-templated mask.
        light_nongermline_col: Column name for light chain non-templated mask.
        heavy_coords_col: Column name for heavy chain coordinates.
        light_coords_col: Column name for light chain coordinates.

    Returns:
        DataLoader instance with weighted sampling.
    """
    datasets = {
        name: AntibodyDataset(
            path,
            max_length=max_length,
            heavy_col=heavy_col,
            light_col=light_col,
            heavy_cdr_col=heavy_cdr_col,
            light_cdr_col=light_cdr_col,
            heavy_nongermline_col=heavy_nongermline_col,
            light_nongermline_col=light_nongermline_col,
            load_coords=load_coords,
            heavy_coords_col=heavy_coords_col,
            light_coords_col=light_coords_col,
        )
        for name, path in data_paths.items()
    }

    multi_dataset = MultiDataset(datasets, weights)

    sampler = WeightedRandomSampler(
        weights=multi_dataset.get_sampler_weights(),
        num_samples=len(multi_dataset),
        replacement=True,
    )

    collator = AntibodyCollator(max_length=max_length, pad_to_max=pad_to_max)

    return DataLoader(
        multi_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator,
    )


def create_train_dataloader(
    cfg: DictConfig,
    batch_size: int,
) -> DataLoader:
    """Create training dataloader from config.

    Automatically selects single or multi-dataset based on train config format.

    Args:
        cfg: Data configuration with `train` key.
        batch_size: Training batch size.

    Returns:
        DataLoader for training.

    Raises:
        ValueError: If data.train is not configured.
    """
    train_cfg = cfg.train

    if train_cfg is None:
        raise ValueError(
            "Training data is required but data.train is not configured.\n"
            "Specify training data via:\n"
            "  - Config file: Set data.train in configs/data/default.yaml\n"
            "  - CLI override: somatic train data.train=/path/to/train.csv\n"
            "  - Multi-dataset: somatic train +data.train.main.path=/path/to/data.csv"
        )

    # Common parameters from config
    common_params = {
        "max_length": cfg.max_length,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": cfg.drop_last,
        "pad_to_max": cfg.pad_to_max,
        "load_coords": cfg.get("load_coords", False),
        "heavy_col": cfg.heavy_col,
        "light_col": cfg.light_col,
        "heavy_cdr_col": cfg.heavy_cdr_col,
        "light_cdr_col": cfg.light_cdr_col,
        "heavy_nongermline_col": cfg.heavy_nongermline_col,
        "light_nongermline_col": cfg.light_nongermline_col,
        "heavy_coords_col": cfg.heavy_coords_col,
        "light_coords_col": cfg.light_coords_col,
    }

    if is_single_train_dataset(train_cfg):
        # Single dataset
        return create_dataloader(
            data_path=train_cfg,
            batch_size=batch_size,
            shuffle=True,
            **common_params,
        )
    else:
        # Multi-dataset
        paths, fractions = parse_train_config(train_cfg)
        return create_multi_dataloader(
            data_paths=paths,
            weights=fractions,
            batch_size=batch_size,
            **common_params,
        )


def create_structure_dataloader(
    folder_path: str | Path,
    batch_size: int,
    max_length: int = 320,
    num_workers: int = 4,
    pin_memory: bool = True,
    chain_id: str | None = None,
    strict: bool = False,
    recursive: bool = False,
) -> DataLoader:
    """Create a DataLoader for structure data (PDB/mmCIF files).

    Args:
        folder_path: Path to folder containing PDB/mmCIF files.
        batch_size: Batch size.
        max_length: Maximum sequence length (for padding/truncation).
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for faster GPU transfer.
        chain_id: Specific chain to extract from each file.
        strict: If True, raise on missing backbone atoms.
        recursive: If True, search subdirectories recursively.

    Returns:
        DataLoader instance for structure data.
    """
    dataset = StructureDataset(
        folder_path=folder_path,
        max_length=max_length,
        chain_id=chain_id,
        strict=strict,
        recursive=recursive,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for eval
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Don't drop last for eval
    )


def create_eval_dataloaders(
    cfg: DictConfig,
    default_batch_size: int,
) -> dict[str, DataLoader]:
    """Create evaluation dataloaders from config.

    Automatically detects dataset format (sequence vs structure) based on the
    path if not explicitly specified. Sequence datasets use AntibodyDataset,
    while structure datasets use StructureDataset.

    Args:
        cfg: Data configuration with `eval` key.
        default_batch_size: Default batch size (used if not specified per-dataset).

    Returns:
        Dict mapping eval dataset names to DataLoaders. Returns empty dict if
        no eval datasets are configured.
    """
    eval_configs = parse_eval_config(cfg.get("eval"), cfg)

    if not eval_configs:
        return {}

    dataloaders: dict[str, DataLoader] = {}

    for name, dataset_cfg in eval_configs.items():
        # Per-dataset overrides with global defaults
        batch_size = dataset_cfg.batch_size or default_batch_size

        # Determine format: explicit or auto-detect
        dataset_format = dataset_cfg.format
        if dataset_format is None:
            dataset_format = detect_dataset_format(dataset_cfg.path)

        if dataset_format == "structure":
            # Structure dataset
            dataloaders[name] = create_structure_dataloader(
                folder_path=dataset_cfg.path,
                batch_size=batch_size,
                max_length=cfg.max_length,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                chain_id=dataset_cfg.chain_id,
                strict=dataset_cfg.strict,
                recursive=dataset_cfg.recursive,
            )
        else:
            # Sequence dataset (default)
            load_coords = (
                dataset_cfg.load_coords
                if dataset_cfg.load_coords is not None
                else cfg.get("load_coords", False)
            )

            dataloaders[name] = create_dataloader(
                data_path=dataset_cfg.path,
                batch_size=batch_size,
                max_length=cfg.max_length,
                shuffle=False,  # No shuffle for eval
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                drop_last=False,  # Don't drop last for eval
                pad_to_max=cfg.pad_to_max,
                load_coords=load_coords,
                heavy_col=cfg.heavy_col,
                light_col=cfg.light_col,
                heavy_cdr_col=cfg.heavy_cdr_col,
                light_cdr_col=cfg.light_cdr_col,
                heavy_nongermline_col=cfg.heavy_nongermline_col,
                light_nongermline_col=cfg.light_nongermline_col,
                heavy_coords_col=cfg.heavy_coords_col,
                light_coords_col=cfg.light_coords_col,
            )

    return dataloaders
