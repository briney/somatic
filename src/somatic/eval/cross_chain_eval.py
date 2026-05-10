"""Cross-chain attention evaluation.

Runs an UNMASKED forward pass with ``output_attentions=True`` and reports
two scalars per dataset:

- ``cross_frac``: fraction of attention mass crossing chain boundaries,
  averaged over heads, layers, and valid (non-pad, non-special) query
  positions.
- ``interface_frac``: of the total cross-chain mass, the fraction
  concentrated in the heavy/light boundary window. Specifically, the
  symmetric sum
  ``mass(last-N heavy -> first-N light) + mass(first-N light -> last-N heavy)``
  divided by the total cross-chain mass.

Mirrors the "separate eval" pattern used by ``region_eval`` — invoked
from ``Evaluator.evaluate()`` after the masked main eval, results are
prefixed with ``"cross_chain/"`` before merging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from ..utils.progress import ProgressManager
from .cross_chain_config import CrossChainEvalConfig
from .region_eval import _get_model_device

if TYPE_CHECKING:
    from accelerate import Accelerator

    from ..model import SomaticModel


def _slice_batch(batch: dict, start: int, end: int) -> dict:
    """Slice all tensor fields of a batch along the batch dimension."""
    return {
        k: (v[start:end] if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _build_masks(
    chain_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    interface_n: int,
) -> dict[str, torch.Tensor]:
    """Construct the per-sub-batch masks used by both metrics.

    All boolean tensors. Shapes:
    - ``valid``: (B, S) — non-pad, non-special query/key positions.
    - ``cross_mask``: (B, S, S) — cross-chain key for any (potentially
      special) query.
    - ``cross_mask_valid_q``: (B, S, S) — same but with the query side
      restricted to non-special.
    - ``interface_pair``: (B, S, S) — boundary window in both directions.
    """
    valid = attention_mask.bool() & ~special_tokens_mask.bool()  # (B, S)

    chain_q = chain_ids.unsqueeze(2)  # (B, S, 1)
    chain_k = chain_ids.unsqueeze(1)  # (B, 1, S)
    cross_pair = chain_q != chain_k  # (B, S, S)

    valid_key = attention_mask.bool().unsqueeze(1)  # (B, 1, S)
    cross_mask = cross_pair & valid_key  # (B, S, S)
    cross_mask_valid_q = cross_mask & valid.unsqueeze(-1)

    heavy_pos = (chain_ids == 0) & valid  # (B, S)
    light_pos = (chain_ids == 1) & valid

    # First-N light: cumulative count along seq dim, accept rows where 1<=count<=N.
    light_cum = torch.cumsum(light_pos.long(), dim=1)
    first_n_light = light_pos & (light_cum >= 1) & (light_cum <= interface_n)

    # Last-N heavy: rank-from-end of heavy positions. total - cumsum + 1
    # gives 1 for the last heavy, 2 for the previous, etc. Multiply by
    # heavy_pos so non-heavy positions are 0 (excluded from <=N test).
    heavy_cum = torch.cumsum(heavy_pos.long(), dim=1)
    total_heavy = heavy_pos.long().sum(dim=1, keepdim=True)
    rank_from_end = (total_heavy - heavy_cum + 1) * heavy_pos.long()
    last_n_heavy = heavy_pos & (rank_from_end >= 1) & (rank_from_end <= interface_n)

    interface_pair = (
        last_n_heavy.unsqueeze(2) & first_n_light.unsqueeze(1)
    ) | (
        first_n_light.unsqueeze(2) & last_n_heavy.unsqueeze(1)
    )  # (B, S, S)

    return {
        "valid": valid,
        "cross_mask": cross_mask,
        "cross_mask_valid_q": cross_mask_valid_q,
        "interface_pair": interface_pair,
    }


def run_cross_chain_eval(
    model: "SomaticModel",
    eval_loader: DataLoader,
    config: CrossChainEvalConfig,
    accelerator: "Accelerator | None",
    show_progress: bool,
    progress: ProgressManager | None = None,
) -> dict[str, float]:
    """Run cross-chain attention evaluation over ``eval_loader``.

    Performs a separate, unmasked forward pass with
    ``output_attentions=True``. Sub-batches each incoming batch by
    ``config.chunk_size`` to bound the memory footprint of holding
    per-layer attention tensors.

    Returns a dict with keys ``cross_frac``, ``interface_frac``, and
    ``interface_n`` (echoed for traceability).
    """
    device = _get_model_device(model, accelerator)

    cross_num = 0.0
    cross_den = 0.0
    if_num = 0.0
    if_den = 0.0

    eval_task_cm = (
        progress.eval_task("Cross-chain attention eval", total=len(eval_loader))
        if progress is not None
        else ProgressManager.standalone_eval_task(
            "Cross-chain attention eval",
            total=len(eval_loader),
            disable=not show_progress,
        )
    )

    model.eval()
    with torch.no_grad(), eval_task_cm as progress_task:
        for batch in eval_loader:
            if batch.get("special_tokens_mask") is None:
                raise ValueError(
                    "cross-chain eval requires 'special_tokens_mask' in the "
                    "batch (the standard AntibodyCollator provides it)."
                )

            if accelerator is None:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            outer_B = batch["token_ids"].shape[0]
            for start in range(0, outer_B, config.chunk_size):
                end = min(start + config.chunk_size, outer_B)
                sub = _slice_batch(batch, start, end)

                outputs = model(
                    token_ids=sub["token_ids"],
                    chain_ids=sub["chain_ids"],
                    attention_mask=sub["attention_mask"],
                    output_attentions=True,
                )
                attentions = outputs["attentions"]  # tuple of (b, H, S, S)

                masks = _build_masks(
                    chain_ids=sub["chain_ids"],
                    attention_mask=sub["attention_mask"],
                    special_tokens_mask=sub["special_tokens_mask"],
                    interface_n=config.interface_n,
                )
                valid = masks["valid"]
                cross_mask = masks["cross_mask"]
                cross_mask_valid_q = masks["cross_mask_valid_q"]
                interface_pair = masks["interface_pair"]

                cross_mask_f = cross_mask.unsqueeze(1).to(attentions[0].dtype)
                cross_mask_vq_f = cross_mask_valid_q.unsqueeze(1).to(attentions[0].dtype)
                interface_pair_f = interface_pair.unsqueeze(1).to(attentions[0].dtype)
                valid_q = valid.unsqueeze(1).to(attentions[0].dtype)

                n_heads = attentions[0].shape[1]
                valid_count = float(valid.sum().item())

                for attn in attentions:
                    # cross_frac numerator: per (B,H,Q-valid) fraction of
                    # attention mass going cross-chain.
                    per_q_cross = (attn * cross_mask_f).sum(dim=-1)  # (b, H, S)
                    cross_num += float((per_q_cross * valid_q).sum().item())
                    # Per softmax row mass is ~1, restricted to valid queries.
                    cross_den += valid_count * n_heads

                    # interface_frac: numerator uses interface_pair (already
                    # implicitly query-restricted); denominator must also be
                    # query-restricted to stay consistent.
                    if_num += float((attn * interface_pair_f).sum().item())
                    if_den += float((attn * cross_mask_vq_f).sum().item())

                del outputs, attentions

            progress_task.advance()

    cross_frac = cross_num / max(cross_den, 1.0)
    interface_frac = if_num / max(if_den, 1e-12)

    return {
        "cross_frac": float(cross_frac),
        "interface_frac": float(interface_frac),
        "interface_n": float(config.interface_n),
    }
