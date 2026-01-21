"""Reproducibility utilities."""

from __future__ import annotations

import os
import random

import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed for reproducibility.

    Parameters
    ----------
    seed
        Random seed value.
    deterministic
        If True, use deterministic algorithms where possible.
        May reduce performance.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def get_generator(seed: int, device: str = "cpu") -> torch.Generator:
    """Create a seeded random generator.

    Parameters
    ----------
    seed
        Random seed value.
    device
        Device for the generator.

    Returns
    -------
    torch.Generator
        Seeded random generator.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
