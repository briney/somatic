"""Configuration for region-specific evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from .regions import AGGREGATE_GROUP_NAMES, INDIVIDUAL_REGION_NAMES


@dataclass
class RegionEvalConfig:
    """Configuration for region-specific evaluation.

    Uses fine-grained boolean flags for each region and aggregate group,
    similar to MaskingFrequencyConfig. This allows maximum flexibility
    in choosing which regions to evaluate and which aggregates to compute.

    Parameters
    ----------
    enabled
        Master switch to enable/disable region evaluation.
    mode
        Evaluation mode: "standard", "per-position", or "region-level".
    position_batch_size
        Batch size for per-position evaluation mode.
    hcdr1, hcdr2, hcdr3
        Enable evaluation for individual heavy chain CDR regions.
    lcdr1, lcdr2, lcdr3
        Enable evaluation for individual light chain CDR regions.
    hfwr1, hfwr2, hfwr3, hfwr4
        Enable evaluation for individual heavy chain framework regions.
    lfwr1, lfwr2, lfwr3, lfwr4
        Enable evaluation for individual light chain framework regions.
    all_cdr
        Enable aggregate statistics for all CDRs combined.
    all_fwr
        Enable aggregate statistics for all frameworks combined.
    heavy
        Enable aggregate statistics for all heavy chain regions.
    light
        Enable aggregate statistics for all light chain regions.
    overall
        Enable aggregate statistics across all regions.
    germline
        Enable aggregate statistics for germline positions (non_templated_mask == 0).
    nongermline
        Enable aggregate statistics for nongermline positions (non_templated_mask == 1).
    """

    enabled: bool = False
    mode: str = "per-position"  # standard | per-position | region-level
    position_batch_size: int = 32

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

    def get_enabled_regions(self) -> set[str]:
        """Return set of individually enabled region names.

        Returns
        -------
        set[str]
            Set of region names (e.g., {"hcdr1", "hcdr3", "lcdr3"}).
        """
        return {r for r in INDIVIDUAL_REGION_NAMES if getattr(self, r, False)}

    def get_enabled_aggregates(self) -> set[str]:
        """Return set of enabled aggregate group names.

        Returns
        -------
        set[str]
            Set of aggregate names (e.g., {"all_cdr", "heavy"}).
        """
        return {a for a in AGGREGATE_GROUP_NAMES if getattr(self, a, False)}

    def has_any_enabled(self) -> bool:
        """Check if any region or aggregate is enabled.

        Returns
        -------
        bool
            True if at least one region or aggregate is enabled.
        """
        return bool(self.get_enabled_regions() or self.get_enabled_aggregates())


def build_region_eval_config(cfg_dict: dict) -> RegionEvalConfig:
    """Build RegionEvalConfig from a dictionary (e.g., from Hydra config).

    Parameters
    ----------
    cfg_dict
        Dictionary with config values. Unknown keys are ignored.

    Returns
    -------
    RegionEvalConfig
        Configured RegionEvalConfig instance.
    """
    if not cfg_dict:
        return RegionEvalConfig()

    # Build config from available fields
    config_kwargs = {}
    for field_name in RegionEvalConfig.__dataclass_fields__:
        if field_name in cfg_dict:
            config_kwargs[field_name] = cfg_dict[field_name]

    return RegionEvalConfig(**config_kwargs)
