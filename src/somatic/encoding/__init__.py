"""Encoding API for extracting embeddings."""

from .encoder import SomaticEncoder
from .pooling import (
    CLSPooling,
    MaxPooling,
    MeanMaxPooling,
    MeanPooling,
    PoolingStrategy,
    PoolingType,
    create_pooling,
)

__all__ = [
    "SomaticEncoder",
    "PoolingStrategy",
    "PoolingType",
    "MeanPooling",
    "CLSPooling",
    "MaxPooling",
    "MeanMaxPooling",
    "create_pooling",
]
