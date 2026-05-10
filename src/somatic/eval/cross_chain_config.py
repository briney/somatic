"""Configuration for cross-chain attention evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossChainEvalConfig:
    """Configuration for cross-chain attention evaluation.

    Parameters
    ----------
    enabled
        Master switch to enable/disable cross-chain attention evaluation.
    interface_n
        Number of tokens at the heavy/light boundary that define the
        "interface" window (last-N heavy positions and first-N light
        positions).
    chunk_size
        Sub-batch size for the unmasked forward pass that captures
        attention weights. The eval loader's batch is sliced into chunks
        of this size to bound peak memory used by ``(B, H, L, L)``
        attention tensors held across all layers.
    """

    enabled: bool = False
    interface_n: int = 5
    chunk_size: int = 8


def build_cross_chain_eval_config(cfg_dict: dict | None) -> CrossChainEvalConfig:
    """Build a CrossChainEvalConfig from a dictionary (e.g., Hydra config).

    Unknown keys are ignored. Raises ``ValueError`` on invalid values.
    """
    if not cfg_dict:
        return CrossChainEvalConfig()

    config_kwargs = {}
    for field_name in CrossChainEvalConfig.__dataclass_fields__:
        if field_name in cfg_dict:
            config_kwargs[field_name] = cfg_dict[field_name]

    config = CrossChainEvalConfig(**config_kwargs)

    if config.interface_n < 1:
        raise ValueError(f"interface_n must be >= 1, got {config.interface_n}")
    if config.chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {config.chunk_size}")

    return config
