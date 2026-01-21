"""Antibody region definitions and extraction utilities."""

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor


class AntibodyRegion(str, Enum):
    """Antibody structural regions.

    Regions follow the standard antibody structure:
    FWR1 - CDR1 - FWR2 - CDR2 - FWR3 - CDR3 - FWR4

    Prefixes indicate chain: h for heavy, l for light.
    Uses immunologically standard naming: hcdr1, lcdr3, hfwr1, etc.
    """

    HCDR1 = "hcdr1"
    HCDR2 = "hcdr2"
    HCDR3 = "hcdr3"
    LCDR1 = "lcdr1"
    LCDR2 = "lcdr2"
    LCDR3 = "lcdr3"
    HFWR1 = "hfwr1"
    HFWR2 = "hfwr2"
    HFWR3 = "hfwr3"
    HFWR4 = "hfwr4"
    LFWR1 = "lfwr1"
    LFWR2 = "lfwr2"
    LFWR3 = "lfwr3"
    LFWR4 = "lfwr4"


# Convenience groupings
CDR_REGIONS = frozenset(
    {
        AntibodyRegion.HCDR1,
        AntibodyRegion.HCDR2,
        AntibodyRegion.HCDR3,
        AntibodyRegion.LCDR1,
        AntibodyRegion.LCDR2,
        AntibodyRegion.LCDR3,
    }
)

FWR_REGIONS = frozenset(
    {
        AntibodyRegion.HFWR1,
        AntibodyRegion.HFWR2,
        AntibodyRegion.HFWR3,
        AntibodyRegion.HFWR4,
        AntibodyRegion.LFWR1,
        AntibodyRegion.LFWR2,
        AntibodyRegion.LFWR3,
        AntibodyRegion.LFWR4,
    }
)

HEAVY_REGIONS = frozenset({r for r in AntibodyRegion if r.value.startswith("h")})
LIGHT_REGIONS = frozenset({r for r in AntibodyRegion if r.value.startswith("l")})

# CDR regions per chain
HEAVY_CDR_REGIONS = frozenset(
    {AntibodyRegion.HCDR1, AntibodyRegion.HCDR2, AntibodyRegion.HCDR3}
)
LIGHT_CDR_REGIONS = frozenset(
    {AntibodyRegion.LCDR1, AntibodyRegion.LCDR2, AntibodyRegion.LCDR3}
)

# Region name constants for iteration (used by config classes)
INDIVIDUAL_REGION_NAMES = (
    "hcdr1",
    "hcdr2",
    "hcdr3",
    "lcdr1",
    "lcdr2",
    "lcdr3",
    "hfwr1",
    "hfwr2",
    "hfwr3",
    "hfwr4",
    "lfwr1",
    "lfwr2",
    "lfwr3",
    "lfwr4",
)

AGGREGATE_GROUP_NAMES = (
    "all_cdr",
    "all_fwr",
    "heavy",
    "light",
    "overall",
    "germline",
    "nongermline",
)


def _find_contiguous_regions(
    mask: Tensor,
    chain_mask: Tensor,
) -> list[tuple[int, int]]:
    """Find contiguous regions of 1s in a 1D mask within a chain.

    Parameters
    ----------
    mask
        1D binary mask tensor.
    chain_mask
        1D binary mask indicating positions belonging to this chain.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) indices for each contiguous region.
        Indices are inclusive start, exclusive end.
    """
    # Apply chain mask
    combined = mask & chain_mask

    regions = []
    in_region = False
    start = 0

    for i in range(len(combined)):
        if combined[i] and not in_region:
            # Start of new region
            in_region = True
            start = i
        elif not combined[i] and in_region:
            # End of region
            in_region = False
            regions.append((start, i))

    # Handle region that extends to end
    if in_region:
        regions.append((start, len(combined)))

    return regions


def _infer_framework_regions(
    cdr_regions: list[tuple[int, int]],
    chain_start: int,
    chain_end: int,
    special_tokens_mask: Tensor | None,
) -> list[tuple[int, int]]:
    """Infer framework regions from CDR boundaries.

    Frameworks are the gaps between CDRs:
    FW1: chain_start to CDR1_start
    FW2: CDR1_end to CDR2_start
    FW3: CDR2_end to CDR3_start
    FW4: CDR3_end to chain_end

    Parameters
    ----------
    cdr_regions
        List of (start, end) for CDR1, CDR2, CDR3 in order.
    chain_start
        Start index of the chain (excluding special tokens).
    chain_end
        End index of the chain (excluding special tokens).
    special_tokens_mask
        Optional mask for special tokens to exclude.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) for FW1, FW2, FW3, FW4.
    """
    if len(cdr_regions) != 3:
        # If we don't have exactly 3 CDRs, return empty frameworks
        return []

    fw_regions = []

    # FW1: chain_start to CDR1_start
    fw1_start = chain_start
    fw1_end = cdr_regions[0][0]
    if fw1_end > fw1_start:
        fw_regions.append((fw1_start, fw1_end))
    else:
        fw_regions.append((fw1_start, fw1_start))  # Empty region

    # FW2: CDR1_end to CDR2_start
    fw2_start = cdr_regions[0][1]
    fw2_end = cdr_regions[1][0]
    if fw2_end > fw2_start:
        fw_regions.append((fw2_start, fw2_end))
    else:
        fw_regions.append((fw2_start, fw2_start))

    # FW3: CDR2_end to CDR3_start
    fw3_start = cdr_regions[1][1]
    fw3_end = cdr_regions[2][0]
    if fw3_end > fw3_start:
        fw_regions.append((fw3_start, fw3_end))
    else:
        fw_regions.append((fw3_start, fw3_start))

    # FW4: CDR3_end to chain_end
    fw4_start = cdr_regions[2][1]
    fw4_end = chain_end
    if fw4_end > fw4_start:
        fw_regions.append((fw4_start, fw4_end))
    else:
        fw_regions.append((fw4_start, fw4_start))

    return fw_regions


def _extract_cdr_boundaries(
    cdr_mask: Tensor,
    chain_mask: Tensor,
) -> list[tuple[int, int]]:
    """Extract CDR region boundaries from detailed CDR mask.

    Uses mask values (1=CDR1, 2=CDR2, 3=CDR3) to find CDR boundaries
    within a single chain.

    Parameters
    ----------
    cdr_mask
        1D tensor with values 0=FW, 1=CDR1, 2=CDR2, 3=CDR3.
    chain_mask
        1D boolean mask indicating positions belonging to this chain.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) indices for CDR1, CDR2, CDR3 in order.
        Indices are inclusive start, exclusive end.
    """
    regions = []

    for cdr_value in [1, 2, 3]:  # CDR1, CDR2, CDR3
        cdr_positions = ((cdr_mask == cdr_value) & chain_mask).nonzero(as_tuple=True)[0]
        if len(cdr_positions) > 0:
            start = cdr_positions[0].item()
            end = cdr_positions[-1].item() + 1
            regions.append((start, end))

    return regions


def extract_region_masks(
    batch: dict[str, Tensor],
    regions: set[AntibodyRegion] | None = None,
) -> dict[AntibodyRegion, Tensor]:
    """Extract per-region boolean masks from batch data.

    Uses detailed cdr_mask values (0=FW, 1=CDR1, 2=CDR2, 3=CDR3) for direct
    region identification. Framework regions are inferred from non-CDR positions.

    The sequence format is assumed to be: [CLS] heavy_chain light_chain [EOS]
    - chain_ids: 0 for CLS and heavy chain, 1 for light chain and EOS
    - cdr_mask: 0 for FW, 1 for CDR1, 2 for CDR2, 3 for CDR3

    Parameters
    ----------
    batch
        Batch dictionary with:
        - cdr_mask: (batch, seq_len) detailed CDR mask (0=FW, 1=CDR1, 2=CDR2, 3=CDR3)
        - chain_ids: (batch, seq_len) chain identifiers (0=heavy, 1=light)
        - attention_mask: (batch, seq_len) valid position mask
        - special_tokens_mask: (batch, seq_len) optional special tokens mask
    regions
        If provided, only extract these regions. Otherwise extract all.

    Returns
    -------
    dict[AntibodyRegion, Tensor]
        Dictionary mapping each region to a (batch, seq_len) boolean mask.
    """
    cdr_mask = batch.get("cdr_mask")
    chain_ids = batch["chain_ids"]
    attention_mask = batch["attention_mask"]
    special_tokens_mask = batch.get("special_tokens_mask")

    if cdr_mask is None:
        raise ValueError("cdr_mask is required to extract region masks")

    batch_size, seq_len = chain_ids.shape
    device = chain_ids.device

    if regions is None:
        regions = set(AntibodyRegion)

    result: dict[AntibodyRegion, Tensor] = {}

    # Initialize all requested region masks to zeros
    for region in regions:
        result[region] = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    # Create chain masks (excluding special tokens if present)
    heavy_chain_mask = (chain_ids == 0) & attention_mask.bool()
    light_chain_mask = (chain_ids == 1) & attention_mask.bool()

    if special_tokens_mask is not None:
        heavy_chain_mask = heavy_chain_mask & ~special_tokens_mask.bool()
        light_chain_mask = light_chain_mask & ~special_tokens_mask.bool()

    # Direct CDR extraction using detailed mask values (vectorized)
    cdr1_mask = cdr_mask == 1
    cdr2_mask = cdr_mask == 2
    cdr3_mask = cdr_mask == 3

    # Assign CDR regions by chain (vectorized across batch)
    if AntibodyRegion.HCDR1 in regions:
        result[AntibodyRegion.HCDR1] = cdr1_mask & heavy_chain_mask
    if AntibodyRegion.HCDR2 in regions:
        result[AntibodyRegion.HCDR2] = cdr2_mask & heavy_chain_mask
    if AntibodyRegion.HCDR3 in regions:
        result[AntibodyRegion.HCDR3] = cdr3_mask & heavy_chain_mask
    if AntibodyRegion.LCDR1 in regions:
        result[AntibodyRegion.LCDR1] = cdr1_mask & light_chain_mask
    if AntibodyRegion.LCDR2 in regions:
        result[AntibodyRegion.LCDR2] = cdr2_mask & light_chain_mask
    if AntibodyRegion.LCDR3 in regions:
        result[AntibodyRegion.LCDR3] = cdr3_mask & light_chain_mask

    # Infer framework regions from CDR boundaries (requires per-sequence processing)
    needs_fwr = any(r in regions for r in FWR_REGIONS)
    if needs_fwr:
        for b in range(batch_size):
            seq_cdr_mask = cdr_mask[b]
            seq_chain_ids = chain_ids[b]
            seq_attention = attention_mask[b].bool()
            seq_special = special_tokens_mask[b] if special_tokens_mask is not None else None

            # Create per-sequence chain masks
            seq_heavy_mask = (seq_chain_ids == 0) & seq_attention
            seq_light_mask = (seq_chain_ids == 1) & seq_attention
            if seq_special is not None:
                seq_heavy_mask = seq_heavy_mask & ~seq_special.bool()
                seq_light_mask = seq_light_mask & ~seq_special.bool()

            # Process heavy chain frameworks
            heavy_positions = seq_heavy_mask.nonzero(as_tuple=True)[0]
            if len(heavy_positions) > 0:
                heavy_start = heavy_positions[0].item()
                heavy_end = heavy_positions[-1].item() + 1

                # Get CDR boundaries from detailed mask
                heavy_cdr_regions = _extract_cdr_boundaries(
                    seq_cdr_mask, seq_heavy_mask
                )
                if len(heavy_cdr_regions) == 3:
                    fwr_regions = _infer_framework_regions(
                        heavy_cdr_regions, heavy_start, heavy_end, seq_special
                    )
                    fwr_names = [
                        AntibodyRegion.HFWR1,
                        AntibodyRegion.HFWR2,
                        AntibodyRegion.HFWR3,
                        AntibodyRegion.HFWR4,
                    ]
                    for i, (start, end) in enumerate(fwr_regions):
                        if fwr_names[i] in regions and end > start:
                            result[fwr_names[i]][b, start:end] = True

            # Process light chain frameworks
            light_positions = seq_light_mask.nonzero(as_tuple=True)[0]
            if len(light_positions) > 0:
                light_start = light_positions[0].item()
                light_end = light_positions[-1].item() + 1

                # Get CDR boundaries from detailed mask
                light_cdr_regions = _extract_cdr_boundaries(
                    seq_cdr_mask, seq_light_mask
                )
                if len(light_cdr_regions) == 3:
                    fwr_regions = _infer_framework_regions(
                        light_cdr_regions, light_start, light_end, seq_special
                    )
                    fwr_names = [
                        AntibodyRegion.LFWR1,
                        AntibodyRegion.LFWR2,
                        AntibodyRegion.LFWR3,
                        AntibodyRegion.LFWR4,
                    ]
                    for i, (start, end) in enumerate(fwr_regions):
                        if fwr_names[i] in regions and end > start:
                            result[fwr_names[i]][b, start:end] = True

    return result


def aggregate_region_masks(
    region_masks: dict[AntibodyRegion, Tensor],
    aggregate_by: str,
) -> dict[str, Tensor]:
    """Aggregate region masks by category.

    Parameters
    ----------
    region_masks
        Dictionary mapping regions to masks.
    aggregate_by
        Aggregation strategy:
        - "all": Return individual regions (no aggregation)
        - "cdr": Aggregate all CDRs together
        - "fw": Aggregate all frameworks together
        - "chain": Aggregate by heavy/light chain
        - "region_type": Aggregate by CDR number (CDR1, CDR2, CDR3) or FW number

    Returns
    -------
    dict[str, Tensor]
        Aggregated masks with descriptive keys.
    """
    result: dict[str, Tensor] = {}

    if aggregate_by == "all":
        # Return individual regions
        for region, mask in region_masks.items():
            result[region.value] = mask

    elif aggregate_by == "cdr":
        # Aggregate all CDRs
        cdr_masks = [m for r, m in region_masks.items() if r in CDR_REGIONS]
        if cdr_masks:
            result["cdr"] = torch.stack(cdr_masks).any(dim=0)

        # Aggregate all frameworks
        fwr_masks = [m for r, m in region_masks.items() if r in FWR_REGIONS]
        if fwr_masks:
            result["fwr"] = torch.stack(fwr_masks).any(dim=0)

    elif aggregate_by == "fwr":
        # Same as "cdr" - aggregate by CDR vs FWR
        cdr_masks = [m for r, m in region_masks.items() if r in CDR_REGIONS]
        if cdr_masks:
            result["cdr"] = torch.stack(cdr_masks).any(dim=0)

        fwr_masks = [m for r, m in region_masks.items() if r in FWR_REGIONS]
        if fwr_masks:
            result["fwr"] = torch.stack(fwr_masks).any(dim=0)

    elif aggregate_by == "chain":
        # Aggregate by chain
        heavy_masks = [m for r, m in region_masks.items() if r in HEAVY_REGIONS]
        if heavy_masks:
            result["heavy"] = torch.stack(heavy_masks).any(dim=0)

        light_masks = [m for r, m in region_masks.items() if r in LIGHT_REGIONS]
        if light_masks:
            result["light"] = torch.stack(light_masks).any(dim=0)

    elif aggregate_by == "region_type":
        # Aggregate by CDR/FWR number across chains
        # CDR1 (heavy + light)
        cdr1_masks = [
            m
            for r, m in region_masks.items()
            if r in {AntibodyRegion.HCDR1, AntibodyRegion.LCDR1}
        ]
        if cdr1_masks:
            result["cdr1"] = torch.stack(cdr1_masks).any(dim=0)

        # CDR2
        cdr2_masks = [
            m
            for r, m in region_masks.items()
            if r in {AntibodyRegion.HCDR2, AntibodyRegion.LCDR2}
        ]
        if cdr2_masks:
            result["cdr2"] = torch.stack(cdr2_masks).any(dim=0)

        # CDR3
        cdr3_masks = [
            m
            for r, m in region_masks.items()
            if r in {AntibodyRegion.HCDR3, AntibodyRegion.LCDR3}
        ]
        if cdr3_masks:
            result["cdr3"] = torch.stack(cdr3_masks).any(dim=0)

        # Frameworks
        for i in range(1, 5):
            fwr_masks = [
                m
                for r, m in region_masks.items()
                if r.value.endswith(f"fwr{i}")
            ]
            if fwr_masks:
                result[f"fwr{i}"] = torch.stack(fwr_masks).any(dim=0)

    else:
        raise ValueError(
            f"Unknown aggregate_by: {aggregate_by}. "
            "Must be one of: 'all', 'cdr', 'fwr', 'chain', 'region_type'"
        )

    return result
