"""Token embeddings for the language model."""

from __future__ import annotations

import math

import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """Token embedding layer with optional scaling."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = 1,
        scale: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = math.sqrt(d_model) if scale else 1.0
        self.d_model = d_model

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embedding(token_ids) * self.scale


class SomaticEmbedding(nn.Module):
    """
    Embedding module for Somatic model.

    Provides token embeddings with optional dropout.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, token_ids: Tensor) -> Tensor:
        embeddings = self.token_embedding(token_ids)
        return self.dropout(embeddings)
