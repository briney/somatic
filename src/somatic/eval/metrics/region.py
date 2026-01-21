"""Per-region metrics for antibody evaluation."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor

from ..base import MetricBase
from ..regions import (
    AntibodyRegion,
    aggregate_region_masks,
    extract_region_masks,
)


class RegionAccuracyMetric(MetricBase):
    """Per-region masked token accuracy.

    Computes accuracy separately for each antibody region, allowing
    detailed analysis of model performance across CDRs and frameworks.

    Only counts positions where both:
    - mask_labels is True (position was masked)
    - region_mask is True (position belongs to region)

    Parameters
    ----------
    regions : list[str] | None
        List of region names to evaluate. If None, evaluates all regions.
    aggregate_by : str
        How to aggregate results: "all", "cdr", "fwr", "chain", "region_type".
    """

    name: ClassVar[str] = "region_acc"
    requires_coords: ClassVar[bool] = False
    needs_attentions: ClassVar[bool] = False

    def __init__(
        self,
        regions: list[str] | None = None,
        aggregate_by: str = "all",
        **kwargs,
    ) -> None:
        super().__init__()
        self.aggregate_by = aggregate_by

        # Parse region names to enum values
        if regions is not None:
            self.regions = {AntibodyRegion(r) for r in regions}
        else:
            self.regions = set(AntibodyRegion)

        # Accumulators per region: {region_name: (correct, total)}
        self._correct: dict[str, int] = {}
        self._total: dict[str, int] = {}
        self._init_accumulators()

    def _init_accumulators(self) -> None:
        """Initialize accumulators for all tracked regions."""
        self._correct = {}
        self._total = {}

        if self.aggregate_by == "all":
            for region in self.regions:
                self._correct[region.value] = 0
                self._total[region.value] = 0
        elif self.aggregate_by in ("cdr", "fwr"):
            self._correct["cdr"] = 0
            self._total["cdr"] = 0
            self._correct["fwr"] = 0
            self._total["fwr"] = 0
        elif self.aggregate_by == "chain":
            self._correct["heavy"] = 0
            self._total["heavy"] = 0
            self._correct["light"] = 0
            self._total["light"] = 0
        elif self.aggregate_by == "region_type":
            for name in ["cdr1", "cdr2", "cdr3", "fwr1", "fwr2", "fwr3", "fwr4"]:
                self._correct[name] = 0
                self._total[name] = 0

    def update(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate accuracy per region from a batch.

        Args:
            outputs: Model outputs with "logits" key.
            batch: Input batch with "token_ids", "cdr_mask", "chain_ids".
            mask_labels: Binary mask indicating masked positions.
        """
        # Skip if no CDR mask available
        if batch.get("cdr_mask") is None:
            return

        logits = outputs["logits"]
        targets = batch["token_ids"]
        predictions = logits.argmax(dim=-1)

        # Get region masks
        try:
            region_masks = extract_region_masks(batch, self.regions)
        except ValueError:
            # No cdr_mask in batch
            return

        # Aggregate if needed
        if self.aggregate_by != "all":
            aggregated = aggregate_region_masks(region_masks, self.aggregate_by)
        else:
            aggregated = {r.value: m for r, m in region_masks.items()}

        mask = mask_labels.bool()
        correct_mask = (predictions == targets) & mask

        for region_name, region_mask in aggregated.items():
            # Only count positions that are both masked and in this region
            combined_mask = mask & region_mask
            region_correct = (correct_mask & region_mask).sum().item()
            region_total = combined_mask.sum().item()

            if region_name in self._correct:
                self._correct[region_name] += region_correct
                self._total[region_name] += region_total

    def compute(self) -> dict[str, float]:
        """Compute per-region accuracy from accumulated counts.

        Returns:
            Dictionary with "{region_name}/acc" keys.
        """
        results = {}
        for region_name in self._correct:
            total = self._total[region_name]
            acc = self._correct[region_name] / total if total > 0 else 0.0
            results[f"{region_name}/acc"] = acc
        return results

    def reset(self) -> None:
        """Reset accumulated state."""
        self._init_accumulators()

    def state_tensors(self) -> list[Tensor]:
        """Return state as tensors for distributed aggregation."""
        # Flatten to [correct1, total1, correct2, total2, ...]
        values = []
        for region_name in sorted(self._correct.keys()):
            values.extend([float(self._correct[region_name]), float(self._total[region_name])])
        return [torch.tensor(values)]

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors and len(tensors) > 0:
            values = tensors[0]
            sorted_keys = sorted(self._correct.keys())
            for i, region_name in enumerate(sorted_keys):
                self._correct[region_name] = int(values[i * 2].item())
                self._total[region_name] = int(values[i * 2 + 1].item())


class RegionPerplexityMetric(MetricBase):
    """Per-region perplexity metric.

    Computes perplexity (exp of cross-entropy loss) separately for each
    antibody region.

    Parameters
    ----------
    regions : list[str] | None
        List of region names to evaluate. If None, evaluates all regions.
    aggregate_by : str
        How to aggregate results: "all", "cdr", "fwr", "chain", "region_type".
    """

    name: ClassVar[str] = "region_ppl"
    requires_coords: ClassVar[bool] = False
    needs_attentions: ClassVar[bool] = False

    def __init__(
        self,
        regions: list[str] | None = None,
        aggregate_by: str = "all",
        **kwargs,
    ) -> None:
        super().__init__()
        self.aggregate_by = aggregate_by

        if regions is not None:
            self.regions = {AntibodyRegion(r) for r in regions}
        else:
            self.regions = set(AntibodyRegion)

        self._total_loss: dict[str, float] = {}
        self._total_tokens: dict[str, int] = {}
        self._init_accumulators()

    def _init_accumulators(self) -> None:
        """Initialize accumulators for all tracked regions."""
        self._total_loss = {}
        self._total_tokens = {}

        if self.aggregate_by == "all":
            for region in self.regions:
                self._total_loss[region.value] = 0.0
                self._total_tokens[region.value] = 0
        elif self.aggregate_by in ("cdr", "fwr"):
            self._total_loss["cdr"] = 0.0
            self._total_tokens["cdr"] = 0
            self._total_loss["fwr"] = 0.0
            self._total_tokens["fwr"] = 0
        elif self.aggregate_by == "chain":
            self._total_loss["heavy"] = 0.0
            self._total_tokens["heavy"] = 0
            self._total_loss["light"] = 0.0
            self._total_tokens["light"] = 0
        elif self.aggregate_by == "region_type":
            for name in ["cdr1", "cdr2", "cdr3", "fw1", "fw2", "fw3", "fw4"]:
                self._total_loss[name] = 0.0
                self._total_tokens[name] = 0

    def update(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate loss per region from a batch."""
        if batch.get("cdr_mask") is None:
            return

        logits = outputs["logits"]
        targets = batch["token_ids"]

        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        loss_per_token = torch.nn.functional.cross_entropy(
            logits_flat, targets_flat, reduction="none"
        ).view(batch_size, seq_len)

        try:
            region_masks = extract_region_masks(batch, self.regions)
        except ValueError:
            return

        if self.aggregate_by != "all":
            aggregated = aggregate_region_masks(region_masks, self.aggregate_by)
        else:
            aggregated = {r.value: m for r, m in region_masks.items()}

        mask = mask_labels.bool()

        for region_name, region_mask in aggregated.items():
            combined_mask = mask & region_mask
            region_loss = (loss_per_token * combined_mask.float()).sum().item()
            region_tokens = combined_mask.sum().item()

            if region_name in self._total_loss:
                self._total_loss[region_name] += region_loss
                self._total_tokens[region_name] += region_tokens

    def compute(self) -> dict[str, float]:
        """Compute per-region perplexity from accumulated loss."""
        results = {}
        for region_name in self._total_loss:
            tokens = self._total_tokens[region_name]
            if tokens > 0:
                avg_loss = self._total_loss[region_name] / tokens
                ppl = torch.exp(torch.tensor(avg_loss)).item()
            else:
                ppl = float("inf")
            results[f"{region_name}/ppl"] = ppl
        return results

    def reset(self) -> None:
        """Reset accumulated state."""
        self._init_accumulators()

    def state_tensors(self) -> list[Tensor]:
        """Return state as tensors for distributed aggregation."""
        values = []
        for region_name in sorted(self._total_loss.keys()):
            values.extend([self._total_loss[region_name], float(self._total_tokens[region_name])])
        return [torch.tensor(values)]

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors and len(tensors) > 0:
            values = tensors[0]
            sorted_keys = sorted(self._total_loss.keys())
            for i, region_name in enumerate(sorted_keys):
                self._total_loss[region_name] = values[i * 2].item()
                self._total_tokens[region_name] = int(values[i * 2 + 1].item())


class RegionLossMetric(MetricBase):
    """Per-region cross-entropy loss metric.

    Computes average cross-entropy loss separately for each antibody region.

    Parameters
    ----------
    regions : list[str] | None
        List of region names to evaluate. If None, evaluates all regions.
    aggregate_by : str
        How to aggregate results: "all", "cdr", "fwr", "chain", "region_type".
    """

    name: ClassVar[str] = "region_loss"
    requires_coords: ClassVar[bool] = False
    needs_attentions: ClassVar[bool] = False

    def __init__(
        self,
        regions: list[str] | None = None,
        aggregate_by: str = "all",
        **kwargs,
    ) -> None:
        super().__init__()
        self.aggregate_by = aggregate_by

        if regions is not None:
            self.regions = {AntibodyRegion(r) for r in regions}
        else:
            self.regions = set(AntibodyRegion)

        self._total_loss: dict[str, float] = {}
        self._total_tokens: dict[str, int] = {}
        self._init_accumulators()

    def _init_accumulators(self) -> None:
        """Initialize accumulators for all tracked regions."""
        self._total_loss = {}
        self._total_tokens = {}

        if self.aggregate_by == "all":
            for region in self.regions:
                self._total_loss[region.value] = 0.0
                self._total_tokens[region.value] = 0
        elif self.aggregate_by in ("cdr", "fwr"):
            self._total_loss["cdr"] = 0.0
            self._total_tokens["cdr"] = 0
            self._total_loss["fwr"] = 0.0
            self._total_tokens["fwr"] = 0
        elif self.aggregate_by == "chain":
            self._total_loss["heavy"] = 0.0
            self._total_tokens["heavy"] = 0
            self._total_loss["light"] = 0.0
            self._total_tokens["light"] = 0
        elif self.aggregate_by == "region_type":
            for name in ["cdr1", "cdr2", "cdr3", "fw1", "fw2", "fw3", "fw4"]:
                self._total_loss[name] = 0.0
                self._total_tokens[name] = 0

    def update(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate loss per region from a batch."""
        if batch.get("cdr_mask") is None:
            return

        logits = outputs["logits"]
        targets = batch["token_ids"]

        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        loss_per_token = torch.nn.functional.cross_entropy(
            logits_flat, targets_flat, reduction="none"
        ).view(batch_size, seq_len)

        try:
            region_masks = extract_region_masks(batch, self.regions)
        except ValueError:
            return

        if self.aggregate_by != "all":
            aggregated = aggregate_region_masks(region_masks, self.aggregate_by)
        else:
            aggregated = {r.value: m for r, m in region_masks.items()}

        mask = mask_labels.bool()

        for region_name, region_mask in aggregated.items():
            combined_mask = mask & region_mask
            region_loss = (loss_per_token * combined_mask.float()).sum().item()
            region_tokens = combined_mask.sum().item()

            if region_name in self._total_loss:
                self._total_loss[region_name] += region_loss
                self._total_tokens[region_name] += region_tokens

    def compute(self) -> dict[str, float]:
        """Compute per-region average loss."""
        results = {}
        for region_name in self._total_loss:
            tokens = self._total_tokens[region_name]
            avg_loss = self._total_loss[region_name] / tokens if tokens > 0 else float("inf")
            results[f"{region_name}/loss"] = avg_loss
        return results

    def reset(self) -> None:
        """Reset accumulated state."""
        self._init_accumulators()

    def state_tensors(self) -> list[Tensor]:
        """Return state as tensors for distributed aggregation."""
        values = []
        for region_name in sorted(self._total_loss.keys()):
            values.extend([self._total_loss[region_name], float(self._total_tokens[region_name])])
        return [torch.tensor(values)]

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        """Load state from gathered tensors."""
        if tensors and len(tensors) > 0:
            values = tensors[0]
            sorted_keys = sorted(self._total_loss.keys())
            for i, region_name in enumerate(sorted_keys):
                self._total_loss[region_name] = values[i * 2].item()
                self._total_tokens[region_name] = int(values[i * 2 + 1].item())
