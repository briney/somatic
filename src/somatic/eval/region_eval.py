"""Region-based evaluation functions.

Contains the three region evaluation modes (standard, per-position,
region-level) plus shared helpers extracted from Evaluator.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .per_position import PerPositionEvaluator, RegionMaskingEvaluator
from .region_config import RegionEvalConfig
from .regions import (
    AntibodyRegion,
    CDR_REGIONS,
    FWR_REGIONS,
    HEAVY_REGIONS,
    LIGHT_REGIONS,
    extract_region_masks,
)

if TYPE_CHECKING:
    from accelerate import Accelerator

    from ..model import SomaticModel
    from .masking import EvalMasker

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_GERMLINE_WARNING = (
    "germline/nongermline region tracking enabled but non_templated_mask "
    "not found in batch. Ensure data has columns matching "
    "heavy_nongermline_col and light_nongermline_col config settings."
)


def _make_accumulator() -> dict[str, float]:
    """Create a fresh region accumulator dict."""
    return {"correct": 0, "total_loss": 0.0, "total_prob": 0.0, "count": 0}


def _compute_individual_region_results(
    region_accumulators: dict[str, dict[str, float]],
    config: RegionEvalConfig,
) -> dict[str, float]:
    """Compute final per-region metrics for individually enabled regions.

    Args:
        region_accumulators: Per-region accumulated metrics.
        config: Region evaluation configuration.

    Returns:
        Dict mapping ``{region_name}/{metric}`` to float values.
    """
    results: dict[str, float] = {}
    enabled_regions = config.get_enabled_regions()
    for region_name, acc in region_accumulators.items():
        if region_name in enabled_regions and acc["count"] > 0:
            avg_loss = acc["total_loss"] / acc["count"]
            results[f"{region_name}/accuracy"] = acc["correct"] / acc["count"]
            results[f"{region_name}/loss"] = avg_loss
            results[f"{region_name}/prob"] = acc["total_prob"] / acc["count"]
            results[f"{region_name}/ppl"] = math.exp(avg_loss)
    return results


def _get_model_device(model: "SomaticModel", accelerator: "Accelerator | None") -> torch.device:
    """Get the device the model is on."""
    if accelerator is not None:
        return accelerator.device
    return next(model.parameters()).device


def _accumulate_positions(
    region_accumulators: dict[str, dict[str, float]],
    key: str,
    positions: list[int],
    per_position_results: dict[int, dict[str, float]],
) -> None:
    """Accumulate per-position metrics into a named region accumulator."""
    if not positions:
        return
    if key not in region_accumulators:
        region_accumulators[key] = _make_accumulator()
    acc = region_accumulators[key]
    for pos in positions:
        if pos in per_position_results:
            metrics = per_position_results[pos]
            acc["correct"] += metrics["correct"]
            acc["total_loss"] += metrics["loss"]
            acc["total_prob"] += metrics["prob"]
            acc["count"] += 1


def _evaluate_masked_group(
    model: "SomaticModel",
    token_ids: torch.Tensor,
    group_mask: torch.Tensor,
    chain_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Mask a group of positions and compute aggregate metrics.

    Args:
        model: The model to evaluate.
        token_ids: Token IDs for a single sample (1-D).
        group_mask: Boolean mask of positions to mask and evaluate (1-D).
        chain_ids: Chain IDs for the sample (1-D).
        attention_mask: Attention mask for the sample (1-D).
        device: Device for tensor creation.

    Returns:
        Dict with correct, total_loss, total_prob, count.
    """
    from ..tokenizer import tokenizer

    masked_ids = token_ids.clone()
    masked_ids[group_mask] = tokenizer.mask_token_id

    outputs = model(
        token_ids=masked_ids.unsqueeze(0),
        chain_ids=chain_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
    )
    logits = outputs["logits"][0]

    pos_indices = group_mask.nonzero(as_tuple=True)[0]
    pos_logits = logits[pos_indices]
    targets = token_ids[pos_indices]

    losses = torch.nn.functional.cross_entropy(pos_logits, targets, reduction="none")
    probs = torch.softmax(pos_logits, dim=-1)
    target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    preds = pos_logits.argmax(dim=-1)

    return {
        "correct": (preds == targets).sum().item(),
        "total_loss": losses.sum().item(),
        "total_prob": target_probs.sum().item(),
        "count": len(pos_indices),
    }


def get_regions_for_aggregates(config: RegionEvalConfig) -> set[AntibodyRegion]:
    """Get all regions needed to compute enabled aggregates.

    Args:
        config: Region evaluation configuration.

    Returns:
        Set of AntibodyRegion values needed for aggregate computation.
    """
    regions_needed: set[AntibodyRegion] = set()
    enabled_aggs = config.get_enabled_aggregates()

    aggregate_to_regions = {
        "all_cdr": CDR_REGIONS,
        "all_fwr": FWR_REGIONS,
        "heavy": HEAVY_REGIONS,
        "light": LIGHT_REGIONS,
        "overall": CDR_REGIONS | FWR_REGIONS,
    }

    for agg_name, region_set in aggregate_to_regions.items():
        if agg_name in enabled_aggs:
            regions_needed |= region_set

    return regions_needed


def compute_aggregate_metrics(
    region_accumulators: dict[str, dict[str, float]],
    config: RegionEvalConfig,
) -> dict[str, float]:
    """Compute aggregate metrics based on enabled aggregates.

    Args:
        region_accumulators: Per-region accumulated metrics.
        config: Region evaluation configuration.

    Returns:
        Aggregated metrics with / separator.
    """

    def _metrics_from_accumulator(
        acc: dict[str, float], prefix: str
    ) -> dict[str, float]:
        """Convert accumulator to metric dict with prefix."""
        if acc["count"] <= 0:
            return {}
        avg_loss = acc["total_loss"] / acc["count"]
        return {
            f"{prefix}/accuracy": acc["correct"] / acc["count"],
            f"{prefix}/loss": avg_loss,
            f"{prefix}/prob": acc["total_prob"] / acc["count"],
            f"{prefix}/ppl": math.exp(avg_loss),
        }

    def _aggregate_regions(
        region_names: set[str], accumulators: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Sum accumulators for specified regions."""
        agg = _make_accumulator()
        for name in region_names:
            if name in accumulators:
                for k in agg:
                    agg[k] += accumulators[name][k]
        return agg

    results: dict[str, float] = {}
    enabled_aggs = config.get_enabled_aggregates()

    aggregate_to_region_names = {
        "all_cdr": {r.value for r in CDR_REGIONS},
        "all_fwr": {r.value for r in FWR_REGIONS},
        "heavy": {r.value for r in HEAVY_REGIONS},
        "light": {r.value for r in LIGHT_REGIONS},
    }

    for agg_name, region_names in aggregate_to_region_names.items():
        if agg_name in enabled_aggs:
            agg_acc = _aggregate_regions(region_names, region_accumulators)
            results.update(_metrics_from_accumulator(agg_acc, agg_name))

    # overall: aggregate all regions (excluding germline/nongermline which are position-based)
    if "overall" in enabled_aggs:
        all_region_names = {
            name for name in region_accumulators if name not in ("germline", "nongermline")
        }
        overall_acc = _aggregate_regions(all_region_names, region_accumulators)
        results.update(_metrics_from_accumulator(overall_acc, "overall"))

    # germline/nongermline: already accumulated directly in region_accumulators
    for position_agg in ("germline", "nongermline"):
        if position_agg in enabled_aggs and position_agg in region_accumulators:
            results.update(
                _metrics_from_accumulator(region_accumulators[position_agg], position_agg)
            )

    return results


# ---------------------------------------------------------------------------
# Standard region evaluation
# ---------------------------------------------------------------------------


def run_standard_eval(
    model: "SomaticModel",
    eval_loader: DataLoader,
    regions: set[AntibodyRegion] | None,
    config: RegionEvalConfig,
    accelerator: "Accelerator | None",
    eval_masker: "EvalMasker | None",
    create_eval_mask,
    show_progress: bool,
) -> dict[str, float]:
    """Run standard region evaluation using existing masking.

    This mode computes region metrics on positions that are both:
    - Masked during evaluation
    - Within the specified regions

    Args:
        model: The model to evaluate.
        eval_loader: DataLoader for the evaluation dataset.
        regions: Set of regions to evaluate, or None for all.
        config: Region evaluation configuration.
        accelerator: Optional Accelerator for distributed training.
        eval_masker: Optional configured evaluation masker.
        create_eval_mask: Callable to create fallback eval masks.
        show_progress: Whether to show a progress bar.

    Returns:
        Dictionary mapping region metric names to values.
    """
    device = _get_model_device(model, accelerator)

    region_accumulators: dict[str, dict[str, float]] = {}

    # Determine which regions to extract for aggregates
    regions_for_aggregates = get_regions_for_aggregates(config)
    all_regions_needed = (regions or set()) | regions_for_aggregates

    # Get seeded generator for reproducible masking
    generator = None
    if eval_masker is not None:
        generator = eval_masker.get_generator(device)

    enabled_aggs = config.get_enabled_aggregates()
    warned_missing_mask = False

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            eval_loader, desc="Region eval (standard)", disable=not show_progress
        ):
            # Move batch to device if not using accelerator
            if accelerator is None:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            # Skip if no CDR mask available
            if batch.get("cdr_mask") is None:
                continue

            # Create mask labels (same logic as main evaluate())
            if eval_masker is not None:
                masked_ids, mask_labels = eval_masker.apply_mask(
                    batch=batch,
                    generator=generator,
                )
            else:
                mask_labels = create_eval_mask(batch, device)
                masked_ids = batch["token_ids"].clone()
                from ..tokenizer import tokenizer

                masked_ids[mask_labels.bool()] = tokenizer.mask_token_id

            # Forward pass
            outputs = model(
                token_ids=masked_ids,
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
            )

            logits = outputs["logits"]
            targets = batch["token_ids"]
            predictions = logits.argmax(dim=-1)

            # Compute per-token loss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            loss_per_token = torch.nn.functional.cross_entropy(
                logits_flat, targets_flat, reduction="none"
            ).view(batch_size, seq_len)

            # Compute per-token probabilities for correct tokens
            probs = torch.softmax(logits, dim=-1)
            target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

            # Extract region masks
            try:
                target_regions = (
                    all_regions_needed if all_regions_needed else set(AntibodyRegion)
                )
                region_masks = extract_region_masks(batch, target_regions)
            except ValueError:
                continue

            mask = mask_labels.bool()
            correct_mask = (predictions == targets) & mask

            # Process individual regions
            for region_name, region_mask in region_masks.items():
                combined_mask = mask & region_mask
                region_correct = (correct_mask & region_mask).sum().item()
                region_loss = (loss_per_token * combined_mask.float()).sum().item()
                region_total = combined_mask.sum().item()

                if region_name.value not in region_accumulators:
                    region_accumulators[region_name.value] = _make_accumulator()
                acc = region_accumulators[region_name.value]
                region_prob = (target_probs * combined_mask.float()).sum().item()
                acc["correct"] += region_correct
                acc["total_loss"] += region_loss
                acc["total_prob"] += region_prob
                acc["count"] += region_total

            # Handle germline/nongermline aggregates (position-based, not region-based)
            non_templated_mask = batch.get("non_templated_mask")
            if non_templated_mask is None:
                if not warned_missing_mask and (
                    "germline" in enabled_aggs or "nongermline" in enabled_aggs
                ):
                    warnings.warn(_GERMLINE_WARNING)
                    warned_missing_mask = True
            if non_templated_mask is not None:
                if "germline" in enabled_aggs:
                    if "germline" not in region_accumulators:
                        region_accumulators["germline"] = _make_accumulator()
                    germline_mask = non_templated_mask == 0
                    combined = mask & germline_mask
                    acc = region_accumulators["germline"]
                    acc["correct"] += ((predictions == targets) & combined).sum().item()
                    acc["total_loss"] += (loss_per_token * combined.float()).sum().item()
                    acc["total_prob"] += (target_probs * combined.float()).sum().item()
                    acc["count"] += combined.sum().item()

                if "nongermline" in enabled_aggs:
                    if "nongermline" not in region_accumulators:
                        region_accumulators["nongermline"] = _make_accumulator()
                    nongermline_mask = non_templated_mask == 1
                    combined = mask & nongermline_mask
                    acc = region_accumulators["nongermline"]
                    acc["correct"] += ((predictions == targets) & combined).sum().item()
                    acc["total_loss"] += (loss_per_token * combined.float()).sum().item()
                    acc["total_prob"] += (target_probs * combined.float()).sum().item()
                    acc["count"] += combined.sum().item()

    # Compute final metrics
    results = _compute_individual_region_results(region_accumulators, config)
    results.update(compute_aggregate_metrics(region_accumulators, config))
    return results


# ---------------------------------------------------------------------------
# Per-position region evaluation
# ---------------------------------------------------------------------------


def run_per_position_eval(
    model: "SomaticModel",
    eval_loader: DataLoader,
    regions: set[AntibodyRegion] | None,
    position_batch_size: int,
    config: RegionEvalConfig,
    accelerator: "Accelerator | None",
    show_progress: bool,
) -> dict[str, float]:
    """Run per-position region evaluation.

    This method is optimized to collect all unique positions needed
    for evaluation upfront, run inference once, and aggregate results
    multiple ways (by region, germline, nongermline).

    Args:
        model: The model to evaluate.
        eval_loader: DataLoader for the evaluation dataset.
        regions: Set of regions to evaluate, or None for all.
        position_batch_size: Batch size for per-position evaluation.
        config: Region evaluation configuration.
        accelerator: Optional Accelerator for distributed training.
        show_progress: Whether to show a progress bar.

    Returns:
        Dictionary mapping region metric names to values.
    """
    device = _get_model_device(model, accelerator)
    evaluator = PerPositionEvaluator(
        model=model,
        position_batch_size=position_batch_size,
        device=device,
        show_progress=False,  # Outer tqdm handles progress
    )

    region_accumulators: dict[str, dict[str, float]] = {}

    # Determine which regions to evaluate for aggregates
    regions_for_aggregates = get_regions_for_aggregates(config)
    all_regions_needed = (regions or set()) | regions_for_aggregates

    # Check if germline/nongermline aggregates are enabled
    enabled_aggs = config.get_enabled_aggregates()
    needs_germline = "germline" in enabled_aggs
    needs_nongermline = "nongermline" in enabled_aggs
    warned_missing_mask = False

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            eval_loader, desc="Region eval (per-position)", disable=not show_progress
        ):
            # Process each sample in the batch individually
            batch_size = batch["token_ids"].shape[0]
            for i in range(batch_size):
                # Extract single sample
                sample = {
                    k: v[i] if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }

                # === OPTIMIZED: Collect all positions upfront ===
                # 1. Extract region masks to get positions per region
                batch_sample = {
                    k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                    for k, v in sample.items()
                }

                try:
                    target_regions = all_regions_needed if all_regions_needed else None
                    region_masks = extract_region_masks(batch_sample, target_regions)
                except (ValueError, KeyError) as e:
                    warnings.warn(f"Region mask extraction failed for sample: {e}")
                    continue

                # Build position-to-region mapping and collect all region positions
                region_positions: dict[str, list[int]] = {}
                all_positions_needed: set[int] = set()

                for region, mask in region_masks.items():
                    positions = mask[0].nonzero(as_tuple=True)[0].tolist()
                    region_positions[region.value] = positions
                    all_positions_needed.update(positions)

                # 2. Extract germline/nongermline positions
                germline_positions: list[int] = []
                nongermline_positions: list[int] = []

                if needs_germline or needs_nongermline:
                    non_templated_mask = sample.get("non_templated_mask")
                    if non_templated_mask is None:
                        if not warned_missing_mask:
                            warnings.warn(_GERMLINE_WARNING)
                            warned_missing_mask = True
                    else:
                        # Get valid positions (exclude special tokens and padding)
                        attention_mask = sample.get("attention_mask")
                        special_tokens_mask = sample.get("special_tokens_mask")
                        valid_mask = (
                            attention_mask.bool() if attention_mask is not None else None
                        )
                        if valid_mask is not None and special_tokens_mask is not None:
                            valid_mask = valid_mask & ~special_tokens_mask.bool()

                        if needs_germline:
                            germline_mask = non_templated_mask == 0
                            if valid_mask is not None:
                                germline_mask = germline_mask & valid_mask
                            germline_positions = germline_mask.nonzero(as_tuple=True)[
                                0
                            ].tolist()
                            all_positions_needed.update(germline_positions)

                        if needs_nongermline:
                            nongermline_mask = non_templated_mask == 1
                            if valid_mask is not None:
                                nongermline_mask = nongermline_mask & valid_mask
                            nongermline_positions = nongermline_mask.nonzero(as_tuple=True)[
                                0
                            ].tolist()
                            all_positions_needed.update(nongermline_positions)

                # Skip if no positions to evaluate
                if not all_positions_needed:
                    continue

                # 3. Evaluate ALL positions ONCE
                try:
                    per_position_results = evaluator.evaluate_positions(
                        sample, list(all_positions_needed)
                    )
                except Exception as e:
                    warnings.warn(f"Per-position evaluation failed for sample: {e}")
                    continue

                # 4. Aggregate by region
                for region_name, positions in region_positions.items():
                    _accumulate_positions(
                        region_accumulators, region_name, positions, per_position_results
                    )

                # 5. Aggregate by germline/nongermline
                _accumulate_positions(
                    region_accumulators, "germline", germline_positions, per_position_results
                )
                _accumulate_positions(
                    region_accumulators, "nongermline", nongermline_positions, per_position_results
                )

    # Compute final metrics
    results = _compute_individual_region_results(region_accumulators, config)
    results.update(compute_aggregate_metrics(region_accumulators, config))
    return results


# ---------------------------------------------------------------------------
# Region-level (full region masking) evaluation
# ---------------------------------------------------------------------------


def run_region_level_eval(
    model: "SomaticModel",
    eval_loader: DataLoader,
    regions: set[AntibodyRegion] | None,
    config: RegionEvalConfig,
    accelerator: "Accelerator | None",
    show_progress: bool,
) -> dict[str, float]:
    """Run region-level (full region masking) evaluation.

    Args:
        model: The model to evaluate.
        eval_loader: DataLoader for the evaluation dataset.
        regions: Set of regions to evaluate, or None for all.
        config: Region evaluation configuration.
        accelerator: Optional Accelerator for distributed training.
        show_progress: Whether to show a progress bar.

    Returns:
        Dictionary mapping region metric names to values.
    """
    device = _get_model_device(model, accelerator)
    evaluator = RegionMaskingEvaluator(model=model, device=device)

    region_accumulators: dict[str, dict[str, float]] = {}

    # Determine which regions to evaluate for aggregates
    regions_for_aggregates = get_regions_for_aggregates(config)
    all_regions_needed = (regions or set()) | regions_for_aggregates

    # Check if germline/nongermline aggregates are enabled
    enabled_aggs = config.get_enabled_aggregates()
    needs_germline = "germline" in enabled_aggs
    needs_nongermline = "nongermline" in enabled_aggs
    warned_missing_mask = False

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            eval_loader, desc="Region eval (region-level)", disable=not show_progress
        ):
            # Process each sample in the batch individually
            batch_size = batch["token_ids"].shape[0]
            for i in range(batch_size):
                # Extract single sample
                sample = {
                    k: v[i] if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }

                # Evaluate all regions
                try:
                    target_regions = all_regions_needed if all_regions_needed else None
                    sample_results = evaluator.evaluate_all_regions(sample, target_regions)
                except Exception as e:
                    warnings.warn(f"Region evaluation failed for sample: {e}")
                    continue

                # Accumulate results
                for region_name, metrics in sample_results.items():
                    if region_name not in region_accumulators:
                        region_accumulators[region_name] = _make_accumulator()
                    acc = region_accumulators[region_name]
                    count = metrics.get("count", 0)
                    if count > 0:
                        acc["correct"] += metrics["accuracy"] * count
                        acc["total_loss"] += metrics["avg_loss"] * count
                        acc["total_prob"] += metrics["avg_prob"] * count
                        acc["count"] += count

                # Handle germline/nongermline aggregates (position-based)
                # For region-level mode, mask all germline/nongermline positions at once
                if needs_germline or needs_nongermline:
                    non_templated_mask = sample.get("non_templated_mask")
                    if non_templated_mask is None:
                        if not warned_missing_mask:
                            warnings.warn(_GERMLINE_WARNING)
                            warned_missing_mask = True
                    else:
                        # Move sample to device
                        sample_on_device = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in sample.items()
                        }
                        token_ids = sample_on_device["token_ids"]
                        chain_ids = sample_on_device["chain_ids"]
                        attention_mask = sample_on_device["attention_mask"]
                        non_templated = sample_on_device["non_templated_mask"]

                        # Get valid positions (exclude special tokens and padding)
                        special_tokens_mask = sample_on_device.get("special_tokens_mask")
                        valid_mask = attention_mask.bool()
                        if special_tokens_mask is not None:
                            valid_mask = valid_mask & ~special_tokens_mask.bool()

                        for group_name, group_flag in (
                            ("germline", needs_germline),
                            ("nongermline", needs_nongermline),
                        ):
                            if not group_flag:
                                continue
                            group_mask = (
                                (non_templated == (0 if group_name == "germline" else 1))
                                & valid_mask
                            )
                            if not group_mask.any():
                                continue
                            try:
                                result = _evaluate_masked_group(
                                    model, token_ids, group_mask,
                                    chain_ids, attention_mask, device,
                                )
                                if group_name not in region_accumulators:
                                    region_accumulators[group_name] = _make_accumulator()
                                acc = region_accumulators[group_name]
                                acc["correct"] += result["correct"]
                                acc["total_loss"] += result["total_loss"]
                                acc["total_prob"] += result["total_prob"]
                                acc["count"] += result["count"]
                            except Exception as e:
                                warnings.warn(f"{group_name.title()} evaluation failed: {e}")

    # Compute final metrics
    results = _compute_individual_region_results(region_accumulators, config)
    results.update(compute_aggregate_metrics(region_accumulators, config))
    return results
