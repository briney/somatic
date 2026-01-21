"""Data augmentation and preprocessing transforms."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any


class Transform(ABC):
    """Abstract base class for data transforms."""

    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        pass


class Compose(Transform):
    """Compose multiple transforms together."""

    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            example = t(example)
        return example


class RandomChainSwap(Transform):
    """Randomly swap heavy and light chains with probability p."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        if random.random() < self.p:
            example = example.copy()
            example["heavy_chain"], example["light_chain"] = (
                example["light_chain"],
                example["heavy_chain"],
            )

            if example.get("heavy_cdr_mask") is not None:
                example["heavy_cdr_mask"], example["light_cdr_mask"] = (
                    example["light_cdr_mask"],
                    example["heavy_cdr_mask"],
                )

            if example.get("heavy_non_templated_mask") is not None:
                example["heavy_non_templated_mask"], example["light_non_templated_mask"] = (
                    example["light_non_templated_mask"],
                    example["heavy_non_templated_mask"],
                )

        return example


class SequenceTruncation(Transform):
    """Truncate sequences that exceed max length."""

    def __init__(self, max_length: int = 320) -> None:
        self.max_length = max_length

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        example = example.copy()

        # Account for CLS and EOS tokens
        max_seq_len = self.max_length - 2

        heavy = example["heavy_chain"]
        light = example["light_chain"]
        total_len = len(heavy) + len(light)

        if total_len > max_seq_len:
            # Truncate proportionally
            ratio = max_seq_len / total_len
            heavy_len = int(len(heavy) * ratio)
            light_len = max_seq_len - heavy_len

            example["heavy_chain"] = heavy[:heavy_len]
            example["light_chain"] = light[:light_len]

            # Truncate masks if present
            if example.get("heavy_cdr_mask") is not None:
                example["heavy_cdr_mask"] = example["heavy_cdr_mask"][:heavy_len]
            if example.get("light_cdr_mask") is not None:
                example["light_cdr_mask"] = example["light_cdr_mask"][:light_len]
            if example.get("heavy_non_templated_mask") is not None:
                example["heavy_non_templated_mask"] = example["heavy_non_templated_mask"][:heavy_len]
            if example.get("light_non_templated_mask") is not None:
                example["light_non_templated_mask"] = example["light_non_templated_mask"][:light_len]

        return example
