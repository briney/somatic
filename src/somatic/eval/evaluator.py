"""Main evaluator class for orchestrating metric computation."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import Metric
from .masking import EvalMasker, create_eval_masker
from .region_config import RegionEvalConfig, build_region_eval_config
from .region_eval import (
    run_per_position_eval,
    run_region_level_eval,
    run_standard_eval,
)
from .regions import AntibodyRegion
from .registry import build_metrics

if TYPE_CHECKING:
    from accelerate import Accelerator
    from omegaconf import DictConfig

    from ..model import SomaticModel


def _get_model_device(model: "SomaticModel", accelerator: "Accelerator | None") -> torch.device:
    """Get the device the model is on.

    Args:
        model: The model to check.
        accelerator: Optional Accelerator instance.

    Returns:
        The device the model parameters are on.
    """
    if accelerator is not None:
        return accelerator.device
    return next(model.parameters()).device


class Evaluator:
    """Orchestrates evaluation metric computation.

    The Evaluator manages metric instantiation, caching, and computation
    for multiple evaluation datasets with potentially different metrics.

    Attributes:
        cfg: Full configuration object.
        model: The model to evaluate.
        accelerator: Optional Accelerator for distributed training.
    """

    def __init__(
        self,
        cfg: "DictConfig",
        model: "SomaticModel",
        accelerator: "Accelerator | None" = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            cfg: Full configuration object.
            model: The model to evaluate.
            accelerator: Optional Accelerate accelerator instance.
        """
        self.cfg = cfg
        self.model = model
        self.accelerator = accelerator

        # Determine global coordinate availability
        data_cfg = cfg.get("data", {})
        self.has_coords = bool(data_cfg.get("load_coords", False))

        # Cache for metrics per eval dataset
        self._metrics_cache: dict[str, list[Metric]] = {}

        # Cache for whether attention weights are needed per eval dataset
        self._needs_attentions_cache: dict[str, bool] = {}

        # Initialize evaluation masker from config (if configured)
        self.eval_masker = self._build_eval_masker()

    def _build_eval_masker(self) -> EvalMasker | None:
        """Build evaluation masker from config if configured.

        Returns
        -------
        EvalMasker or None
            Configured EvalMasker if eval.masking is present in config,
            None otherwise.
        """
        eval_cfg = self.cfg.get("eval", {})
        masking_cfg = eval_cfg.get("masking", {})

        if not masking_cfg:
            return None

        return create_eval_masker(masking_cfg)

    def _get_metrics(self, eval_name: str) -> list[Metric]:
        """Get or build metrics for an evaluation dataset.

        Args:
            eval_name: Name of the evaluation dataset.

        Returns:
            List of Metric instances for this dataset.
        """
        if eval_name not in self._metrics_cache:
            metrics = build_metrics(
                cfg=self.cfg,
                has_coords=self.has_coords,
                eval_name=eval_name,
            )
            self._metrics_cache[eval_name] = metrics
        return self._metrics_cache[eval_name]

    def _needs_attentions(self, eval_name: str) -> bool:
        """Check if any metrics need attention weights.

        Args:
            eval_name: Name of the evaluation dataset.

        Returns:
            True if any metric requires attention weights.
        """
        if eval_name not in self._needs_attentions_cache:
            metrics = self._get_metrics(eval_name)
            self._needs_attentions_cache[eval_name] = any(
                getattr(m, "needs_attentions", False) for m in metrics
            )
        return self._needs_attentions_cache[eval_name]

    def _gather_metric_states(self, metrics: list[Metric]) -> None:
        """Aggregate metric states across distributed processes.

        Args:
            metrics: List of metrics to aggregate.
        """
        if self.accelerator is None or self.accelerator.num_processes <= 1:
            return

        for metric in metrics:
            # Check if this metric uses object-based gathering
            state_objects = metric.state_objects()

            if state_objects is not None:
                # Use gather_for_metrics with use_gather_object for variable-length state
                gathered = self.accelerator.gather_for_metrics(
                    state_objects, use_gather_object=True
                )
                metric.load_state_objects(gathered)
            else:
                # Use tensor-based gathering for fixed-size state
                state_tensors = metric.state_tensors()
                if state_tensors:
                    gathered_tensors = []
                    for tensor in state_tensors:
                        # Move to accelerator device and gather
                        tensor = tensor.to(self.accelerator.device)
                        gathered = self.accelerator.gather(tensor)
                        # Sum across processes
                        if gathered.dim() > tensor.dim():
                            gathered = gathered.sum(dim=0)
                        gathered_tensors.append(gathered)
                    metric.load_state_tensors(gathered_tensors)

    def evaluate(
        self,
        eval_loader: DataLoader,
        eval_name: str,
        masker: Any = None,
    ) -> dict[str, float]:
        """Run evaluation on a dataset.

        Masking priority:
        1. Use self.eval_masker if configured (controlled, reproducible eval)
        2. Use passed masker parameter (legacy behavior)
        3. Fall back to _create_eval_mask (15% random masking)

        Args:
            eval_loader: DataLoader for the evaluation dataset.
            eval_name: Name of the evaluation dataset.
            masker: Optional masker for creating mask labels.
                Ignored if self.eval_masker is configured.

        Returns:
            Dictionary mapping metric names to values.
        """
        metrics = self._get_metrics(eval_name)

        if not metrics:
            return {}

        # Reset all metrics
        for metric in metrics:
            metric.reset()

        # Check if any metrics need attention weights
        needs_attentions = self._needs_attentions(eval_name)

        self.model.eval()
        device = _get_model_device(self.model, self.accelerator)

        # Determine if we should show progress bar
        show_progress = self.accelerator is None or self.accelerator.is_local_main_process

        # Get seeded generator for reproducible masking (if using eval_masker)
        generator = None
        if self.eval_masker is not None:
            generator = self.eval_masker.get_generator(device)

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Eval ({eval_name})", disable=not show_progress):
                # Move batch to device if not using accelerator
                if self.accelerator is None:
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                # Create mask labels for evaluation
                # Priority: 1) eval_masker (controlled), 2) passed masker, 3) fallback
                if self.eval_masker is not None:
                    # Use configured eval masker with seeded generator
                    masked_ids, mask_labels = self.eval_masker.apply_mask(
                        batch=batch,
                        generator=generator,
                    )
                elif masker is not None:
                    # Legacy: use passed masker
                    batch_size = batch["token_ids"].shape[0]
                    timesteps = masker.noise_schedule.sample_timesteps(batch_size, device)
                    masked_ids, mask_labels = masker.apply_mask(
                        token_ids=batch["token_ids"],
                        timesteps=timesteps,
                        attention_mask=batch["attention_mask"],
                        special_tokens_mask=batch.get("special_tokens_mask"),
                    )
                else:
                    # Default: random 15% masking for eval
                    mask_labels = self._create_eval_mask(batch, device)
                    masked_ids = batch["token_ids"].clone()
                    from ..tokenizer import tokenizer

                    masked_ids[mask_labels.bool()] = tokenizer.mask_token_id

                # Forward pass
                outputs = self.model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                    output_attentions=needs_attentions,
                )

                # Update all metrics
                for metric in metrics:
                    try:
                        metric.update(outputs, batch, mask_labels)
                    except Exception as e:
                        warnings.warn(f"Metric '{metric.name}' update failed: {e}")

        # Aggregate across distributed processes
        self._gather_metric_states(metrics)

        # Compute final metric values
        results: dict[str, float] = {}
        for metric in metrics:
            try:
                computed = metric.compute()
                results.update(computed)
            except Exception as e:
                warnings.warn(f"Metric '{metric.name}' compute failed: {e}")

        # Region-based evaluation (if enabled for this dataset)
        region_cfg = self._get_region_config(eval_name)
        if region_cfg.get("enabled", False):
            try:
                region_results = self._evaluate_regions(eval_loader, eval_name, region_cfg)
                # Prefix with "region/" and merge into results
                for key, value in region_results.items():
                    results[f"region/{key}"] = value
            except Exception as e:
                warnings.warn(f"Region evaluation failed: {e}")

        self.model.train()

        # Clear CUDA cache to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def _create_eval_mask(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device,
        mask_ratio: float = 0.15,
    ) -> torch.Tensor:
        """Create a random mask for evaluation.

        Args:
            batch: Input batch dictionary.
            device: Device to create the mask on.
            mask_ratio: Fraction of tokens to mask.

        Returns:
            Binary mask tensor (batch, seq_len).
        """
        token_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        special_tokens_mask = batch.get("special_tokens_mask")

        # Random mask
        rand = torch.rand_like(token_ids.float())
        mask_labels = (rand < mask_ratio).long()

        # Don't mask padding
        mask_labels = mask_labels * attention_mask

        # Don't mask special tokens if mask is provided
        if special_tokens_mask is not None:
            mask_labels = mask_labels * (~special_tokens_mask).long()

        return mask_labels.to(device)

    def _get_region_config(self, eval_name: str) -> dict[str, Any]:
        """Get merged region evaluation config for a dataset.

        Merges global eval.regions config with per-dataset overrides.

        Args:
            eval_name: Name of the evaluation dataset.

        Returns:
            Merged region configuration dictionary.
        """
        # Start with global config
        eval_cfg = self.cfg.get("eval", {})
        global_regions = dict(eval_cfg.get("regions", {}))

        # Get per-dataset override if present
        data_cfg = self.cfg.get("data", {})
        eval_datasets = data_cfg.get("eval", {})

        if isinstance(eval_datasets, str):
            # Single eval dataset, no per-dataset config
            return global_regions

        dataset_cfg = eval_datasets.get(eval_name, {})
        if isinstance(dataset_cfg, str):
            # Shorthand path, no per-dataset config
            return global_regions

        dataset_regions = dict(dataset_cfg.get("regions", {}))

        # Merge: dataset overrides global
        result = global_regions.copy()
        result.update(dataset_regions)
        return result

    def _show_progress(self) -> bool:
        """Check if progress bars should be shown."""
        return self.accelerator is None or self.accelerator.is_local_main_process

    def _evaluate_regions(
        self,
        eval_loader: DataLoader,
        eval_name: str,
        region_cfg: dict[str, Any],
        mask_labels_cache: list[torch.Tensor] | None = None,
        batch_cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, float]:
        """Run region-based evaluation.

        Args:
            eval_loader: DataLoader for the evaluation dataset.
            eval_name: Name of the evaluation dataset.
            region_cfg: Region evaluation configuration dictionary.
            mask_labels_cache: Cached mask labels from main evaluation (for standard mode).
            batch_cache: Cached batches from main evaluation (for standard mode).

        Returns:
            Dictionary mapping region metric names to values.
        """
        # Build config object from dictionary
        config = build_region_eval_config(region_cfg)

        mode = config.mode
        position_batch_size = config.position_batch_size

        # Get enabled regions as AntibodyRegion set
        enabled_region_names = config.get_enabled_regions()
        if enabled_region_names:
            regions = {AntibodyRegion(r) for r in enabled_region_names}
        else:
            regions = None  # No individual regions enabled

        show_progress = self._show_progress()

        if mode == "standard":
            return run_standard_eval(
                model=self.model,
                eval_loader=eval_loader,
                regions=regions,
                config=config,
                accelerator=self.accelerator,
                eval_masker=self.eval_masker,
                create_eval_mask=self._create_eval_mask,
                show_progress=show_progress,
            )
        elif mode in ("per_position", "per-position"):
            return run_per_position_eval(
                model=self.model,
                eval_loader=eval_loader,
                regions=regions,
                position_batch_size=position_batch_size,
                config=config,
                accelerator=self.accelerator,
                show_progress=show_progress,
            )
        else:  # region_level or region-level
            return run_region_level_eval(
                model=self.model,
                eval_loader=eval_loader,
                regions=regions,
                config=config,
                accelerator=self.accelerator,
                show_progress=show_progress,
            )

    def evaluate_all(
        self,
        eval_loaders: dict[str, DataLoader],
        masker: Any = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate on all configured evaluation datasets.

        Args:
            eval_loaders: Dictionary mapping eval dataset names to DataLoaders.
            masker: Optional masker for creating mask labels.

        Returns:
            Dictionary mapping eval dataset names to their metric results.
        """
        all_results: dict[str, dict[str, float]] = {}

        for eval_name, eval_loader in eval_loaders.items():
            results = self.evaluate(eval_loader, eval_name, masker)
            all_results[eval_name] = results

        return all_results

    def clear_cache(self) -> None:
        """Clear cached metrics and attention flags.

        Call this if the configuration changes during training.
        """
        self._metrics_cache.clear()
        self._needs_attentions_cache.clear()
