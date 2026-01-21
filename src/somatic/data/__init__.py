"""Data loading components."""

from .collator import AntibodyCollator
from .config import (
    DatasetConfig,
    is_single_eval_dataset,
    is_single_train_dataset,
    normalize_fractions,
    parse_eval_config,
    parse_train_config,
)
from .dataset import (
    SEQUENCE_EXTENSIONS,
    STRUCTURE_EXTENSIONS,
    AntibodyDataset,
    MultiDataset,
    StructureData,
    StructureDataset,
    detect_dataset_format,
    parse_structure,
)
from .loader import (
    create_dataloader,
    create_eval_dataloaders,
    create_multi_dataloader,
    create_structure_dataloader,
    create_train_dataloader,
)
from .transforms import Compose, RandomChainSwap, SequenceTruncation, Transform

__all__ = [
    # Dataset classes
    "AntibodyDataset",
    "MultiDataset",
    "StructureDataset",
    # Structure parsing
    "StructureData",
    "parse_structure",
    "detect_dataset_format",
    "STRUCTURE_EXTENSIONS",
    "SEQUENCE_EXTENSIONS",
    # Collator
    "AntibodyCollator",
    # DataLoader factories
    "create_dataloader",
    "create_multi_dataloader",
    "create_train_dataloader",
    "create_eval_dataloaders",
    "create_structure_dataloader",
    # Config
    "DatasetConfig",
    "parse_train_config",
    "parse_eval_config",
    "normalize_fractions",
    "is_single_train_dataset",
    "is_single_eval_dataset",
    # Transforms
    "Transform",
    "Compose",
    "RandomChainSwap",
    "SequenceTruncation",
]
