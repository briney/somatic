"""Per-position and region-level evaluation for antibody analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from tqdm import tqdm

from ..tokenizer import tokenizer
from .regions import AntibodyRegion, extract_region_masks

if TYPE_CHECKING:
    from ..model import SomaticModel


class PerPositionEvaluator:
    """Evaluate model by masking one position at a time.

    For efficiency, batches positions across the batch dimension:
    - For N positions with position_batch_size=32, runs ceil(N/32) forward passes
    - Each batch item is the same sequence with a different position masked

    This allows detailed per-position analysis and region-level aggregation.

    Parameters
    ----------
    model
        The model to evaluate.
    position_batch_size
        Number of positions to evaluate per forward pass.
    device
        Device to run evaluation on.
    show_progress
        Whether to show progress bar.
    """

    def __init__(
        self,
        model: SomaticModel,
        position_batch_size: int = 32,
        device: torch.device | None = None,
        show_progress: bool = True,
    ) -> None:
        self.model = model
        self.position_batch_size = position_batch_size
        self.device = device or next(model.parameters()).device
        self.show_progress = show_progress

    def evaluate_positions(
        self,
        sample: dict[str, Tensor],
        positions: list[int] | None = None,
    ) -> dict[int, dict[str, float]]:
        """Evaluate each position by masking only that position.

        Parameters
        ----------
        sample
            Single sequence dictionary with:
            - token_ids: (seq_len,) token IDs
            - chain_ids: (seq_len,) chain identifiers
            - attention_mask: (seq_len,) valid position mask
            - special_tokens_mask: (seq_len,) optional special tokens mask

        positions
            List of positions to evaluate. If None, evaluates all valid positions.

        Returns
        -------
        dict[int, dict[str, float]]
            Dictionary mapping position index to metrics dict with keys:
            - "correct": 1 if prediction matches target, 0 otherwise
            - "loss": cross-entropy loss at this position
            - "prob": probability assigned to correct token
        """
        # Move sample to device
        sample = {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in sample.items()
        }

        token_ids = sample["token_ids"]
        chain_ids = sample["chain_ids"]
        attention_mask = sample["attention_mask"]
        special_tokens_mask = sample.get("special_tokens_mask")

        # Determine valid positions
        if positions is None:
            valid = attention_mask.bool()
            if special_tokens_mask is not None:
                valid = valid & ~special_tokens_mask.bool()
            positions = valid.nonzero(as_tuple=True)[0].tolist()

        results: dict[int, dict[str, float]] = {}

        self.model.eval()
        with torch.no_grad():
            # Process positions in batches
            iterator = range(0, len(positions), self.position_batch_size)
            if self.show_progress:
                iterator = tqdm(iterator, desc="Per-position eval")

            for batch_start in iterator:
                batch_positions = positions[batch_start : batch_start + self.position_batch_size]
                batch_size = len(batch_positions)

                # Create batched input: same sequence repeated batch_size times
                batched_token_ids = token_ids.unsqueeze(0).expand(batch_size, -1).clone()
                batched_chain_ids = chain_ids.unsqueeze(0).expand(batch_size, -1)
                batched_attention = attention_mask.unsqueeze(0).expand(batch_size, -1)

                # Mask one position per batch item
                for i, pos in enumerate(batch_positions):
                    batched_token_ids[i, pos] = tokenizer.mask_token_id

                # Forward pass
                outputs = self.model(
                    token_ids=batched_token_ids,
                    chain_ids=batched_chain_ids,
                    attention_mask=batched_attention,
                )

                logits = outputs["logits"]

                # Extract metrics for each position
                for i, pos in enumerate(batch_positions):
                    pos_logits = logits[i, pos]  # (vocab_size,)
                    target = token_ids[pos].item()

                    # Prediction
                    pred = pos_logits.argmax().item()
                    correct = 1 if pred == target else 0

                    # Loss
                    loss = torch.nn.functional.cross_entropy(
                        pos_logits.unsqueeze(0),
                        torch.tensor([target], device=self.device),
                    ).item()

                    # Probability
                    probs = torch.softmax(pos_logits, dim=-1)
                    prob = probs[target].item()

                    results[pos] = {
                        "correct": correct,
                        "loss": loss,
                        "prob": prob,
                    }

        return results

    def evaluate_by_region(
        self,
        sample: dict[str, Tensor],
        regions: set[AntibodyRegion] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate per-position and aggregate by region.

        Parameters
        ----------
        sample
            Single sequence dictionary with required fields.
        regions
            Regions to evaluate. If None, evaluates all regions.

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary mapping region name to aggregated metrics:
            - "accuracy": fraction of correct predictions
            - "avg_loss": average cross-entropy loss
            - "avg_prob": average probability of correct token
            - "count": number of positions in region
        """
        # Need batch dimension for extract_region_masks
        batch = {
            k: v.unsqueeze(0) if isinstance(v, Tensor) else v
            for k, v in sample.items()
        }

        # Extract region masks
        region_masks = extract_region_masks(batch, regions)

        # Get all positions across all regions
        all_positions = set()
        for region_mask in region_masks.values():
            positions = region_mask[0].nonzero(as_tuple=True)[0].tolist()
            all_positions.update(positions)

        if not all_positions:
            return {}

        # Evaluate all positions
        per_position_results = self.evaluate_positions(sample, list(all_positions))

        # Aggregate by region
        results: dict[str, dict[str, float]] = {}
        for region, region_mask in region_masks.items():
            positions = region_mask[0].nonzero(as_tuple=True)[0].tolist()
            if not positions:
                continue

            total_correct = 0
            total_loss = 0.0
            total_prob = 0.0
            count = 0

            for pos in positions:
                if pos in per_position_results:
                    metrics = per_position_results[pos]
                    total_correct += metrics["correct"]
                    total_loss += metrics["loss"]
                    total_prob += metrics["prob"]
                    count += 1

            if count > 0:
                results[region.value] = {
                    "accuracy": total_correct / count,
                    "avg_loss": total_loss / count,
                    "avg_prob": total_prob / count,
                    "count": count,
                }

        return results


class RegionMaskingEvaluator:
    """Evaluate by masking entire regions at once.

    This tests the model's ability to reconstruct complete regions
    given context from other parts of the sequence.

    Parameters
    ----------
    model
        The model to evaluate.
    device
        Device to run evaluation on.
    """

    def __init__(
        self,
        model: SomaticModel,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device

    def evaluate_region(
        self,
        sample: dict[str, Tensor],
        target_region: AntibodyRegion,
    ) -> dict[str, float]:
        """Evaluate reconstruction of a single region when fully masked.

        Parameters
        ----------
        sample
            Single sequence dictionary with required fields.
        target_region
            The region to mask and evaluate.

        Returns
        -------
        dict[str, float]
            Metrics for reconstructing the masked region:
            - "accuracy": fraction of positions correctly predicted
            - "avg_loss": average cross-entropy loss
            - "avg_prob": average probability of correct tokens
            - "count": number of positions in region
        """
        # Move sample to device
        sample = {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in sample.items()
        }

        # Need batch dimension for extract_region_masks
        batch = {
            k: v.unsqueeze(0) if isinstance(v, Tensor) else v
            for k, v in sample.items()
        }

        # Extract target region mask
        region_masks = extract_region_masks(batch, {target_region})
        if target_region not in region_masks:
            return {}

        region_mask = region_masks[target_region][0]  # Remove batch dim
        positions = region_mask.nonzero(as_tuple=True)[0].tolist()

        if not positions:
            return {}

        token_ids = sample["token_ids"]
        chain_ids = sample["chain_ids"]
        attention_mask = sample["attention_mask"]

        # Create masked version with entire region masked
        masked_ids = token_ids.clone()
        masked_ids[region_mask] = tokenizer.mask_token_id

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                token_ids=masked_ids.unsqueeze(0),
                chain_ids=chain_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
            )

            logits = outputs["logits"][0]  # Remove batch dim

            # Compute metrics for all positions in region
            total_correct = 0
            total_loss = 0.0
            total_prob = 0.0

            for pos in positions:
                pos_logits = logits[pos]
                target = token_ids[pos].item()

                # Prediction
                pred = pos_logits.argmax().item()
                total_correct += 1 if pred == target else 0

                # Loss
                loss = torch.nn.functional.cross_entropy(
                    pos_logits.unsqueeze(0),
                    torch.tensor([target], device=self.device),
                ).item()
                total_loss += loss

                # Probability
                probs = torch.softmax(pos_logits, dim=-1)
                total_prob += probs[target].item()

        count = len(positions)
        return {
            "accuracy": total_correct / count,
            "avg_loss": total_loss / count,
            "avg_prob": total_prob / count,
            "count": count,
        }

    def evaluate_all_regions(
        self,
        sample: dict[str, Tensor],
        regions: set[AntibodyRegion] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate reconstruction of all regions.

        Parameters
        ----------
        sample
            Single sequence dictionary with required fields.
        regions
            Regions to evaluate. If None, evaluates all regions.

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary mapping region name to metrics.
        """
        if regions is None:
            regions = set(AntibodyRegion)

        results: dict[str, dict[str, float]] = {}
        for region in regions:
            metrics = self.evaluate_region(sample, region)
            if metrics:
                results[region.value] = metrics

        return results
