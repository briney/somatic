"""Masking frequency tracking for training and evaluation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

from ..eval.regions import AGGREGATE_GROUP_NAMES, INDIVIDUAL_REGION_NAMES

if TYPE_CHECKING:
    from ..eval.regions import AntibodyRegion


@dataclass
class MaskingFrequencyConfig:
    """Configuration for masking frequency tracking.

    Allows fine-grained control over which regions and aggregates to track.
    Individual regions and aggregate groups can be enabled independently.

    Parameters
    ----------
    enabled
        Master switch to enable/disable tracking.
    hcdr1, hcdr2, hcdr3
        Track individual heavy chain CDR regions.
    lcdr1, lcdr2, lcdr3
        Track individual light chain CDR regions.
    hfwr1, hfwr2, hfwr3, hfwr4
        Track individual heavy chain framework regions.
    lfwr1, lfwr2, lfwr3, lfwr4
        Track individual light chain framework regions.
    all_cdr
        Track all CDRs combined as a single aggregate.
    all_fwr
        Track all frameworks combined as a single aggregate.
    heavy
        Track all heavy chain regions combined.
    light
        Track all light chain regions combined.
    overall
        Track overall masking statistics across all positions.
    germline
        Track masking statistics for germline positions (non_templated_mask == 0).
    nongermline
        Track masking statistics for nongermline positions (non_templated_mask == 1).
    """

    enabled: bool = False

    # Individual CDR regions (6 total)
    hcdr1: bool = False
    hcdr2: bool = False
    hcdr3: bool = False
    lcdr1: bool = False
    lcdr2: bool = False
    lcdr3: bool = False

    # Individual framework regions (8 total)
    hfwr1: bool = False
    hfwr2: bool = False
    hfwr3: bool = False
    hfwr4: bool = False
    lfwr1: bool = False
    lfwr2: bool = False
    lfwr3: bool = False
    lfwr4: bool = False

    # Aggregate groups (7 total)
    all_cdr: bool = False
    all_fwr: bool = False
    heavy: bool = False
    light: bool = False
    overall: bool = False
    germline: bool = False
    nongermline: bool = False


class MaskingFrequencyTracker:
    """Tracks masking frequency per antibody region.

    Computes two metrics per region:
    1. fraction_masked: % of positions in region that were masked
    2. share_of_total: % of all masked tokens that came from this region

    Parameters
    ----------
    config
        Configuration specifying which regions/aggregates to track.

    Examples
    --------
    >>> config = MaskingFrequencyConfig(enabled=True, hcdr1=True, all_cdr=True)
    >>> tracker = MaskingFrequencyTracker(config)
    >>> tracker.update(mask_labels, batch)
    >>> metrics = tracker.compute()
    >>> # {'hcdr1/fraction_masked': 0.3, 'hcdr1/share_of_total': 0.15, ...}
    """

    def __init__(self, config: MaskingFrequencyConfig) -> None:
        self.config = config
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        """Reset all accumulator dictionaries."""
        # Per-region counts: {region_name: {"masked": int, "total": int}}
        self._region_counts: dict[str, dict[str, int]] = {}
        # Global masked count for share_of_total calculation
        self._total_masked: int = 0

    def get_enabled_regions(self) -> set[str]:
        """Return set of individually enabled region names."""
        return {r for r in INDIVIDUAL_REGION_NAMES if getattr(self.config, r, False)}

    def get_enabled_aggregates(self) -> set[str]:
        """Return set of enabled aggregate group names."""
        return {a for a in AGGREGATE_GROUP_NAMES if getattr(self.config, a, False)}

    def update(
        self,
        mask_labels: Tensor,
        batch: dict[str, Tensor],
    ) -> None:
        """Update tracking with a batch of masking results.

        Parameters
        ----------
        mask_labels
            Boolean tensor (batch, seq_len) indicating masked positions.
        batch
            Batch dictionary containing cdr_mask, chain_ids, attention_mask,
            and optionally special_tokens_mask.
        """
        if not self.config.enabled:
            return

        cdr_mask = batch.get("cdr_mask")
        if cdr_mask is None:
            # Graceful degradation - can't track regions without CDR annotations
            return

        # Import here to avoid circular imports
        from ..eval.regions import (
            CDR_REGIONS,
            FWR_REGIONS,
            HEAVY_REGIONS,
            LIGHT_REGIONS,
            AntibodyRegion,
            extract_region_masks,
        )

        enabled_regions = self.get_enabled_regions()
        enabled_aggregates = self.get_enabled_aggregates()

        # Build set of AntibodyRegion enums to extract
        regions_to_extract: set[AntibodyRegion] = set()

        # Add individually enabled regions
        for region_name in enabled_regions:
            regions_to_extract.add(AntibodyRegion(region_name))

        # Add component regions needed for aggregates
        if "all_cdr" in enabled_aggregates:
            regions_to_extract.update(CDR_REGIONS)
        if "all_fwr" in enabled_aggregates:
            regions_to_extract.update(FWR_REGIONS)
        if "heavy" in enabled_aggregates:
            regions_to_extract.update(HEAVY_REGIONS)
        if "light" in enabled_aggregates:
            regions_to_extract.update(LIGHT_REGIONS)

        # Prepare masks
        mask_bool = mask_labels.bool()
        attention_mask = batch["attention_mask"].bool()
        special_tokens_mask = batch.get("special_tokens_mask")

        if special_tokens_mask is not None:
            valid_mask = attention_mask & ~special_tokens_mask.bool()
        else:
            valid_mask = attention_mask

        # Update overall if enabled (doesn't need region extraction)
        if "overall" in enabled_aggregates:
            if "overall" not in self._region_counts:
                self._region_counts["overall"] = {"masked": 0, "total": 0}
            overall_masked = valid_mask & mask_bool
            self._region_counts["overall"]["masked"] += overall_masked.sum().item()
            self._region_counts["overall"]["total"] += valid_mask.sum().item()

        # Track total masked for share_of_total calculation
        self._total_masked += (valid_mask & mask_bool).sum().item()

        # Handle germline/nongermline tracking (position-based, not region-based)
        non_templated_mask = batch.get("non_templated_mask")
        if non_templated_mask is None:
            # Warn once if germline/nongermline tracking enabled but mask missing
            if "germline" in enabled_aggregates or "nongermline" in enabled_aggregates:
                warnings.warn(
                    "germline/nongermline masking frequency tracking enabled but "
                    "non_templated_mask not found in batch. Ensure data has columns matching "
                    "heavy_nongermline_col and light_nongermline_col config settings."
                )
        if non_templated_mask is not None:
            if "germline" in enabled_aggregates:
                if "germline" not in self._region_counts:
                    self._region_counts["germline"] = {"masked": 0, "total": 0}
                germline_mask = (non_templated_mask == 0) & valid_mask
                germline_masked = germline_mask & mask_bool
                self._region_counts["germline"]["masked"] += germline_masked.sum().item()
                self._region_counts["germline"]["total"] += germline_mask.sum().item()

            if "nongermline" in enabled_aggregates:
                if "nongermline" not in self._region_counts:
                    self._region_counts["nongermline"] = {"masked": 0, "total": 0}
                nongermline_mask = (non_templated_mask == 1) & valid_mask
                nongermline_masked = nongermline_mask & mask_bool
                self._region_counts["nongermline"]["masked"] += nongermline_masked.sum().item()
                self._region_counts["nongermline"]["total"] += nongermline_mask.sum().item()

        # Early exit if no regions need extraction
        if not regions_to_extract:
            return

        # Extract region masks
        try:
            region_masks = extract_region_masks(batch, regions_to_extract)
        except ValueError:
            # CDR mask present but invalid format
            return

        # Update individual region counts
        for region, region_mask in region_masks.items():
            region_name = region.value
            if region_name in enabled_regions:
                if region_name not in self._region_counts:
                    self._region_counts[region_name] = {"masked": 0, "total": 0}

                # Positions in this region (valid, non-special)
                region_valid = region_mask & valid_mask
                # Masked positions in this region
                region_masked = region_valid & mask_bool

                self._region_counts[region_name]["masked"] += region_masked.sum().item()
                self._region_counts[region_name]["total"] += region_valid.sum().item()

        # Update aggregate counts
        aggregate_mapping = {
            "all_cdr": CDR_REGIONS,
            "all_fwr": FWR_REGIONS,
            "heavy": HEAVY_REGIONS,
            "light": LIGHT_REGIONS,
        }

        for agg_name, component_regions in aggregate_mapping.items():
            if agg_name not in enabled_aggregates:
                continue

            if agg_name not in self._region_counts:
                self._region_counts[agg_name] = {"masked": 0, "total": 0}

            for region in component_regions:
                if region in region_masks:
                    region_valid = region_masks[region] & valid_mask
                    region_masked = region_valid & mask_bool
                    self._region_counts[agg_name]["masked"] += region_masked.sum().item()
                    self._region_counts[agg_name]["total"] += region_valid.sum().item()

    def compute(self) -> dict[str, float]:
        """Compute final metrics for logging.

        Returns
        -------
        dict[str, float]
            Dictionary with keys like "hcdr1/fraction_masked", "hcdr1/share_of_total".
            Only includes metrics for enabled regions/aggregates that have data.
        """
        if not self.config.enabled:
            return {}

        results: dict[str, float] = {}

        enabled_regions = self.get_enabled_regions()
        enabled_aggregates = self.get_enabled_aggregates()
        all_enabled = enabled_regions | enabled_aggregates

        for name in all_enabled:
            if name not in self._region_counts:
                continue

            counts = self._region_counts[name]

            # Fraction of this region that was masked
            if counts["total"] > 0:
                results[f"{name}/fraction_masked"] = counts["masked"] / counts["total"]

            # Share of total masked tokens from this region
            if self._total_masked > 0:
                results[f"{name}/share_of_total"] = counts["masked"] / self._total_masked

        return results

    def reset(self) -> None:
        """Reset all accumulators for next logging interval."""
        self._reset_accumulators()
