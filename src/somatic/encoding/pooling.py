"""Pooling strategies for sequence embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import Tensor


class PoolingType(str, Enum):
    """Available pooling strategies."""

    MEAN = "mean"
    CLS = "cls"
    MAX = "max"
    MEAN_MAX = "mean_max"


class PoolingStrategy(ABC):
    """Abstract base class for pooling strategies."""

    @abstractmethod
    def __call__(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Apply pooling to hidden states.

        Parameters
        ----------
        hidden_states
            Tensor of shape (batch_size, seq_len, hidden_dim).
        attention_mask
            Optional mask of shape (batch_size, seq_len).

        Returns
        -------
        Tensor
            Pooled embeddings of shape (batch_size, output_dim).
        """
        pass


class MeanPooling(PoolingStrategy):
    """Average pooling over sequence positions."""

    def __call__(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (hidden_states * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


class CLSPooling(PoolingStrategy):
    """Use the CLS token (first position) as the embedding."""

    def __call__(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        return hidden_states[:, 0, :]


class MaxPooling(PoolingStrategy):
    """Max pooling over sequence positions."""

    def __call__(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        if attention_mask is None:
            return hidden_states.max(dim=1).values

        mask = attention_mask.unsqueeze(-1).bool()
        masked_hidden = hidden_states.masked_fill(~mask, float("-inf"))
        return masked_hidden.max(dim=1).values


class MeanMaxPooling(PoolingStrategy):
    """Concatenation of mean and max pooling."""

    def __init__(self) -> None:
        self.mean_pool = MeanPooling()
        self.max_pool = MaxPooling()

    def __call__(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        mean_out = self.mean_pool(hidden_states, attention_mask)
        max_out = self.max_pool(hidden_states, attention_mask)
        return torch.cat([mean_out, max_out], dim=-1)


def create_pooling(pooling_type: str | PoolingType) -> PoolingStrategy:
    """Create a pooling strategy from a type string or enum.

    Parameters
    ----------
    pooling_type
        The type of pooling to use.

    Returns
    -------
    PoolingStrategy
        The pooling strategy instance.

    Raises
    ------
    ValueError
        If the pooling type is unknown.
    """
    if isinstance(pooling_type, str):
        pooling_type = PoolingType(pooling_type.lower())

    if pooling_type == PoolingType.MEAN:
        return MeanPooling()
    elif pooling_type == PoolingType.CLS:
        return CLSPooling()
    elif pooling_type == PoolingType.MAX:
        return MaxPooling()
    elif pooling_type == PoolingType.MEAN_MAX:
        return MeanMaxPooling()
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")
