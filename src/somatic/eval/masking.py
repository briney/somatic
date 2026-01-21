"""Controlled masking for evaluation with seeded reproducibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..masking import InformationWeightedMasker, UniformMasker
from ..tokenizer import tokenizer

if TYPE_CHECKING:
    from omegaconf import DictConfig


class EvalMasker:
    """Controlled masking for evaluation with seeded reproducibility.

    Wraps UniformMasker or InformationWeightedMasker with:
    - Configurable masking strategy (uniform vs information-weighted)
    - Configurable mask rate
    - Seeded random generation for reproducibility

    Parameters
    ----------
    masker_type
        Type of masking: "uniform" or "information_weighted".
    mask_rate
        Target mask rate (0.0-1.0).
    cdr_weight_multiplier
        Weight multiplier for CDR positions in information-weighted masking.
    nongermline_weight_multiplier
        Weight multiplier for nongermline positions in information-weighted masking.
    seed
        Random seed for reproducibility.
    selection_method
        Selection method for information-weighted masking: "ranked" or "sampled".
    """

    def __init__(
        self,
        masker_type: str = "uniform",
        mask_rate: float = 0.15,
        cdr_weight_multiplier: float = 1.0,
        nongermline_weight_multiplier: float = 1.0,
        seed: int = 42,
        selection_method: str = "sampled",
    ) -> None:
        self.masker_type = masker_type
        self.mask_rate = mask_rate
        self.cdr_weight_multiplier = cdr_weight_multiplier
        self.nongermline_weight_multiplier = nongermline_weight_multiplier
        self.seed = seed
        self.selection_method = selection_method

        # Create underlying masker
        if masker_type == "uniform":
            self._masker = UniformMasker(
                mask_rate=mask_rate,
                mask_token_id=tokenizer.mask_token_id,
            )
        elif masker_type == "information_weighted":
            self._masker = InformationWeightedMasker(
                mask_rate=mask_rate,
                cdr_weight_multiplier=cdr_weight_multiplier,
                nongermline_weight_multiplier=nongermline_weight_multiplier,
                mask_token_id=tokenizer.mask_token_id,
                selection_method=selection_method,
            )
        else:
            raise ValueError(
                f"Unknown masker_type: {masker_type}. "
                "Must be 'uniform' or 'information_weighted'."
            )

    def get_generator(self, device: torch.device) -> torch.Generator:
        """Create a fresh seeded generator for an evaluation run.

        Creates a new generator with the configured seed each time,
        ensuring reproducible masking when called at the start of each
        evaluation run.

        Parameters
        ----------
        device
            Device to create the generator on.

        Returns
        -------
        torch.Generator
            Seeded generator for this device.
        """
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)
        return gen

    def apply_mask(
        self,
        batch: dict[str, Tensor],
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply masking to a batch with optional seeded generator.

        Parameters
        ----------
        batch
            Batch dictionary with at least:
            - token_ids: (batch, seq_len) token IDs
            - attention_mask: (batch, seq_len) attention mask
            - special_tokens_mask: (batch, seq_len) optional special tokens mask
            - cdr_mask: (batch, seq_len) optional CDR mask (for info-weighted)
            - non_templated_mask: (batch, seq_len) optional non-templated mask
        generator
            Optional seeded generator for reproducibility.

        Returns
        -------
        tuple[Tensor, Tensor]
            - masked_ids: Token IDs with masked positions replaced
            - mask_labels: Boolean mask indicating which positions were masked
        """
        token_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        special_tokens_mask = batch.get("special_tokens_mask")

        # Apply masking based on masker type
        if self.masker_type == "uniform":
            return self._masker.apply_mask(
                token_ids=token_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                generator=generator,
            )
        else:
            # Information-weighted masking
            return self._masker.apply_mask(
                token_ids=token_ids,
                attention_mask=attention_mask,
                cdr_mask=batch.get("cdr_mask"),
                non_templated_mask=batch.get("non_templated_mask"),
                special_tokens_mask=special_tokens_mask,
                generator=generator,
            )


def create_eval_masker(cfg: DictConfig) -> EvalMasker:
    """Create an EvalMasker from configuration.

    Parameters
    ----------
    cfg
        Configuration with keys:
        - type: "uniform" or "information_weighted"
        - mask_rate: float
        - cdr_weight_multiplier: float (for information_weighted)
        - nongermline_weight_multiplier: float (for information_weighted)
        - seed: int
        - selection_method: "ranked" or "sampled" (for information_weighted)

    Returns
    -------
    EvalMasker
        Configured evaluation masker.
    """
    return EvalMasker(
        masker_type=cfg.get("type", "uniform"),
        mask_rate=cfg.get("mask_rate", 0.15),
        cdr_weight_multiplier=cfg.get("cdr_weight_multiplier", 1.0),
        nongermline_weight_multiplier=cfg.get("nongermline_weight_multiplier", 1.0),
        seed=cfg.get("seed", 42),
        selection_method=cfg.get("selection_method", "sampled"),
    )
