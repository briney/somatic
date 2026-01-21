"""Masking strategies for MLM training."""

from __future__ import annotations

import torch
from torch import Tensor

from ..tokenizer import tokenizer


class InformationWeightedMasker:
    """
    Applies masking with preference for high-information positions.

    CDR and nongermline positions receive higher masking weights, controlled by
    separate multipliers. With default multipliers (1.0 each):
    - Non-templated CDR positions: 3x base weight (1 + 1 + 1)
    - Templated CDR or non-templated non-CDR: 2x base weight
    - Templated non-CDR (framework): 1x base weight

    Two selection methods are available:
    - "ranked": Deterministically masks the top-K highest-weighted positions
    - "sampled": Probabilistically samples positions using Gumbel-top-k,
                 where higher weights increase selection probability but
                 don't guarantee selection
    """

    def __init__(
        self,
        mask_rate: float = 0.15,
        cdr_weight_multiplier: float = 1.0,
        nongermline_weight_multiplier: float = 1.0,
        mask_token_id: int = tokenizer.mask_token_id,
        selection_method: str = "sampled",
    ) -> None:
        if not 0.0 < mask_rate < 1.0:
            raise ValueError(f"mask_rate must be in (0, 1), got {mask_rate}")
        self.mask_rate = mask_rate
        self.cdr_weight_multiplier = cdr_weight_multiplier
        self.nongermline_weight_multiplier = nongermline_weight_multiplier
        self.mask_token_id = mask_token_id
        if selection_method not in ("ranked", "sampled"):
            raise ValueError(
                f"selection_method must be 'ranked' or 'sampled', got {selection_method}"
            )
        self.selection_method = selection_method

    def compute_weights(
        self,
        cdr_mask: Tensor | None,
        non_templated_mask: Tensor | None,
        attention_mask: Tensor,
    ) -> Tensor:
        batch_size, seq_len = attention_mask.shape
        device = attention_mask.device

        weights = torch.ones(batch_size, seq_len, device=device)

        if cdr_mask is not None:
            # Convert detailed CDR mask (0=FW, 1=CDR1, 2=CDR2, 3=CDR3) to binary
            cdr_binary = (cdr_mask > 0).float()
            weights = weights + cdr_binary * self.cdr_weight_multiplier

        if non_templated_mask is not None:
            weights = weights + non_templated_mask.float() * self.nongermline_weight_multiplier

        weights = weights * attention_mask.float()
        weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return weights / weights_sum

    def apply_mask(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
        cdr_mask: Tensor | None = None,
        non_templated_mask: Tensor | None = None,
        special_tokens_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        valid_counts = attention_mask.sum(dim=-1)

        if special_tokens_mask is not None:
            special_counts = (special_tokens_mask & attention_mask.bool()).sum(dim=-1)
            valid_counts = valid_counts - special_counts

        num_to_mask = (valid_counts.float() * self.mask_rate).round().long().clamp(min=0)

        maskable_positions = attention_mask.bool().clone()
        if special_tokens_mask is not None:
            maskable_positions = maskable_positions & ~special_tokens_mask.bool()

        weights = self.compute_weights(cdr_mask, non_templated_mask, maskable_positions)

        # Compute scores based on selection method
        if self.selection_method == "ranked":
            # Deterministic top-K selection (small noise only for tie-breaking)
            noise = torch.rand(weights.shape, device=device, generator=generator) * 1e-6
            scores = weights + noise
        else:
            # Gumbel-top-k: weighted probabilistic sampling without replacement
            # Adding Gumbel noise to log-weights gives proper weighted sampling
            # See: https://arxiv.org/abs/1903.06059 (Gumbel-Top-k trick)
            eps = 1e-10
            uniform = torch.rand(weights.shape, device=device, generator=generator)
            uniform = uniform.clamp(min=eps, max=1 - eps)
            gumbel_noise = -torch.log(-torch.log(uniform))
            scores = torch.log(weights + eps) + gumbel_noise

        scores = scores.masked_fill(~maskable_positions.bool(), float("-inf"))

        _, indices = scores.sort(dim=-1, descending=True)

        position_ranks = torch.zeros_like(indices)
        position_ranks.scatter_(
            dim=-1,
            index=indices,
            src=torch.arange(seq_len, device=device).expand(batch_size, -1),
        )

        mask_labels = position_ranks < num_to_mask.unsqueeze(-1)
        mask_labels = mask_labels & maskable_positions.bool()

        masked_ids = token_ids.clone()
        masked_ids[mask_labels] = self.mask_token_id

        return masked_ids, mask_labels


class UniformMasker:
    """Simple uniform random masking without information weighting."""

    def __init__(
        self,
        mask_rate: float = 0.15,
        mask_token_id: int = tokenizer.mask_token_id,
    ) -> None:
        if not 0.0 < mask_rate < 1.0:
            raise ValueError(f"mask_rate must be in (0, 1), got {mask_rate}")
        self.mask_rate = mask_rate
        self.mask_token_id = mask_token_id

    def apply_mask(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
        special_tokens_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        rand = torch.rand(batch_size, seq_len, device=device, generator=generator)

        maskable = attention_mask.bool()
        if special_tokens_mask is not None:
            maskable = maskable & ~special_tokens_mask.bool()

        mask_labels = (rand < self.mask_rate) & maskable

        masked_ids = token_ids.clone()
        masked_ids[mask_labels] = self.mask_token_id

        return masked_ids, mask_labels
