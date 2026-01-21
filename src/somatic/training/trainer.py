"""Main training loop with Accelerate integration."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..masking import InformationWeightedMasker, UniformMasker
from ..model import SomaticModel
from .checkpoint import CheckpointConfig, CheckpointManager
from .flops import FLOPsConfig, FLOPsTracker
from .masking_frequency import MaskingFrequencyConfig, MaskingFrequencyTracker
from .metrics import (
    MLMMetrics,
    MetricAccumulator,
    compute_masked_cross_entropy,
    compute_mlm_metrics,
)
from .optimizer import create_optimizer, create_scheduler, get_lr

if TYPE_CHECKING:
    from ..eval import Evaluator


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Duration (step-driven by default)
    max_steps: int = 100000
    max_epochs: int | None = None

    # Batch size
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler_decay: str = "cosine"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1

    # Masking
    mask_rate: float = 0.15
    use_information_weighted_masking: bool = True
    cdr_weight_multiplier: float = 1.0
    nongermline_weight_multiplier: float = 1.0
    masking_selection: str = "sampled"  # "ranked" | "sampled"

    # Intervals (in steps)
    log_steps: int = 10
    eval_steps: int = 500
    checkpoint_steps: int = 1000

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 5
    save_best: bool = True

    # Reproducibility
    seed: int = 42

    # Mixed precision
    mixed_precision: str = "no"


class Trainer:
    """Main trainer class with Accelerate integration."""

    def __init__(
        self,
        config: TrainingConfig,
        model: SomaticModel,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        eval_dataloaders: dict[str, DataLoader] | None = None,
        evaluator: "Evaluator | None" = None,
        accelerator: Accelerator | None = None,
        masking_frequency_config: MaskingFrequencyConfig | None = None,
        flops_config: FLOPsConfig | None = None,
    ) -> None:
        self.config = config

        # Use provided accelerator or create a new one
        if accelerator is not None:
            self.accelerator = accelerator
        else:
            # Let accelerate handle mixed_precision from its config (accelerate config)
            # rather than overriding with our config value
            self.accelerator = Accelerator(
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )

        self.optimizer = create_optimizer(
            model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_decay=config.scheduler_decay,
            num_training_steps=config.max_steps,
            num_warmup_steps=config.warmup_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

        (
            self.model,
            self.optimizer,
            self.train_dataloader,
        ) = self.accelerator.prepare(model, self.optimizer, train_dataloader)
        # Note: scheduler is intentionally NOT prepared by Accelerate.
        # AcceleratedScheduler causes 8x step rate in multi-GPU DDP training.

        # Support both single eval_dataloader (legacy) and multiple eval_dataloaders
        self.eval_dataloader = (
            self.accelerator.prepare(eval_dataloader) if eval_dataloader else None
        )

        # Prepare multiple eval dataloaders if provided
        self.eval_dataloaders: dict[str, DataLoader] = {}
        if eval_dataloaders:
            for name, loader in eval_dataloaders.items():
                self.eval_dataloaders[name] = self.accelerator.prepare(loader)
        elif self.eval_dataloader is not None:
            # Use single eval_dataloader as "validation" if no multi-loader dict provided
            self.eval_dataloaders["validation"] = self.eval_dataloader

        # Store evaluator for advanced metrics
        self.evaluator = evaluator

        # Initialize maskers with mask_rate directly
        self.masker = InformationWeightedMasker(
            mask_rate=config.mask_rate,
            cdr_weight_multiplier=config.cdr_weight_multiplier,
            nongermline_weight_multiplier=config.nongermline_weight_multiplier,
            selection_method=config.masking_selection,
        )
        self.uniform_masker = UniformMasker(mask_rate=config.mask_rate)

        checkpoint_config = CheckpointConfig(
            save_dir=config.checkpoint_dir,
            checkpoint_steps=config.checkpoint_steps,
            keep_last_n=config.keep_last_n_checkpoints,
            save_best=config.save_best,
        )
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_config,
            unwrapped_model,
            self.optimizer,
            self.scheduler,
            model_config=unwrapped_model.config,
        )

        self.metrics = MetricAccumulator()
        self.global_step = 0
        self.epoch = 0.0
        self.steps_per_epoch = len(self.train_dataloader)
        self.logger = None

        # Compute total steps for progress tracking
        if config.max_epochs is not None:
            self.total_steps = config.max_epochs * self.steps_per_epoch
        else:
            self.total_steps = config.max_steps

        # Masking frequency tracking
        self.masking_frequency_config = masking_frequency_config or MaskingFrequencyConfig()
        self.masking_frequency_tracker = MaskingFrequencyTracker(self.masking_frequency_config)
        self.eval_masking_frequency_trackers: dict[str, MaskingFrequencyTracker] = {}

        # FLOPs tracking
        self.flops_config = flops_config or FLOPsConfig()
        self.flops_tracker = FLOPsTracker(
            config=self.flops_config,
            model_config=self.accelerator.unwrap_model(self.model).config,
            world_size=self.accelerator.num_processes,
        )

    def set_logger(self, logger) -> None:
        """Set the logger for training metrics."""
        self.logger = logger

    def set_evaluator(self, evaluator: "Evaluator") -> None:
        """Set the evaluator for advanced metrics.

        Args:
            evaluator: Evaluator instance for computing metrics.
        """
        self.evaluator = evaluator

    def _apply_masking(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply masking to a batch."""
        if self.config.use_information_weighted_masking and (
            batch.get("cdr_mask") is not None or batch.get("non_templated_mask") is not None
        ):
            masked_ids, mask_labels = self.masker.apply_mask(
                token_ids=batch["token_ids"],
                attention_mask=batch["attention_mask"],
                cdr_mask=batch.get("cdr_mask"),
                non_templated_mask=batch.get("non_templated_mask"),
                special_tokens_mask=batch.get("special_tokens_mask"),
            )
        else:
            masked_ids, mask_labels = self.uniform_masker.apply_mask(
                token_ids=batch["token_ids"],
                attention_mask=batch["attention_mask"],
                special_tokens_mask=batch.get("special_tokens_mask"),
            )

        return {"masked_ids": masked_ids, "mask_labels": mask_labels}

    def training_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, MLMMetrics]:
        """Execute a single training step.

        Returns:
            Tuple of (loss tensor for backprop, MLMMetrics with all metrics).
        """
        mask_output = self._apply_masking(batch)

        # Track masking frequency
        self.masking_frequency_tracker.update(mask_output["mask_labels"], batch)

        outputs = self.model(
            token_ids=mask_output["masked_ids"],
            chain_ids=batch["chain_ids"],
            attention_mask=batch["attention_mask"],
        )

        metrics = compute_mlm_metrics(
            logits=outputs["logits"],
            targets=batch["token_ids"],
            mask_labels=mask_output["mask_labels"],
            attention_mask=batch["attention_mask"],
        )

        # Compute loss tensor for backprop (metrics.loss is a float)
        loss = compute_masked_cross_entropy(
            logits=outputs["logits"],
            targets=batch["token_ids"],
            mask_labels=mask_output["mask_labels"],
        )

        return loss, metrics

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the eval dataset (legacy single-dataset method)."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        eval_metrics = MetricAccumulator()

        for batch in tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
        ):
            mask_output = self._apply_masking(batch)

            outputs = self.model(
                token_ids=mask_output["masked_ids"],
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
            )

            metrics = compute_mlm_metrics(
                logits=outputs["logits"],
                targets=batch["token_ids"],
                mask_labels=mask_output["mask_labels"],
                attention_mask=batch["attention_mask"],
            )

            eval_metrics.update("loss", metrics.loss)
            eval_metrics.update("accuracy", metrics.accuracy)
            eval_metrics.update("perplexity", metrics.perplexity)

        self.model.train()

        return {
            "val_loss": eval_metrics.compute("loss"),
            "val_accuracy": eval_metrics.compute("accuracy"),
            "val_perplexity": eval_metrics.compute("perplexity"),
        }

    def evaluate_all(self) -> dict[str, dict[str, float]]:
        """Run evaluation on all configured eval datasets.

        Uses the Evaluator if available for advanced metrics, otherwise
        falls back to basic metrics.

        Returns:
            Dictionary mapping eval dataset names to their metric results.
        """
        if not self.eval_dataloaders:
            return {}

        if self.evaluator is not None:
            # Use advanced evaluator
            # Note: masking frequency tracking is handled in simple eval path only.
            # Advanced evaluator does its own masking internally.
            return self.evaluator.evaluate_all(
                self.eval_dataloaders,
                masker=self.uniform_masker,
            )

        # Fall back to simple evaluation for each dataset
        all_results: dict[str, dict[str, float]] = {}
        self.model.eval()

        for eval_name, eval_loader in self.eval_dataloaders.items():
            eval_metrics = MetricAccumulator()

            # Get or create masking frequency tracker for this eval dataset
            if eval_name not in self.eval_masking_frequency_trackers:
                self.eval_masking_frequency_trackers[eval_name] = MaskingFrequencyTracker(
                    self.masking_frequency_config
                )
            eval_tracker = self.eval_masking_frequency_trackers[eval_name]
            eval_tracker.reset()

            for batch in tqdm(
                eval_loader,
                desc=f"Eval ({eval_name})",
                disable=not self.accelerator.is_local_main_process,
            ):
                mask_output = self._apply_masking(batch)

                # Track masking frequency for eval
                eval_tracker.update(mask_output["mask_labels"], batch)

                outputs = self.model(
                    token_ids=mask_output["masked_ids"],
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                metrics = compute_mlm_metrics(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_output["mask_labels"],
                    attention_mask=batch["attention_mask"],
                )

                eval_metrics.update("loss", metrics.loss)
                eval_metrics.update("accuracy", metrics.accuracy)
                eval_metrics.update("perplexity", metrics.perplexity)

            all_results[eval_name] = {
                "loss": eval_metrics.compute("loss"),
                "accuracy": eval_metrics.compute("accuracy"),
                "perplexity": eval_metrics.compute("perplexity"),
            }

            # Add masking frequency metrics
            masking_freq = eval_tracker.compute()
            for key, value in masking_freq.items():
                all_results[eval_name][f"masking_frequency/{key}"] = value

        self.model.train()
        return all_results

    def train(self) -> None:
        """Run the training loop."""
        self.model.train()

        if self.config.max_epochs is not None:
            steps_per_epoch = len(self.train_dataloader)
            total_steps = self.config.max_epochs * steps_per_epoch
        else:
            total_steps = self.config.max_steps

        self.accelerator.print(f"Starting training for {total_steps} steps...")

        progress_bar = tqdm(
            total=total_steps,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
            file=sys.stdout,  # Explicit stdout for proper flushing with accelerate
        )
        progress_bar.update(self.global_step)

        while self.global_step < total_steps:
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    loss, step_metrics = self.training_step(batch)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    # Step scheduler once per actual training step, not per GPU
                    self.scheduler.step()
                    self.global_step += 1
                    self.epoch = self.global_step / self.steps_per_epoch
                    progress_bar.update(1)

                    self.metrics.update("train/loss", step_metrics.loss)
                    self.metrics.update("train/accuracy", step_metrics.accuracy)
                    self.metrics.update("train/perplexity", step_metrics.perplexity)

                    # Update FLOPs tracking
                    batch_size = batch["token_ids"].shape[0]
                    seq_len = batch["token_ids"].shape[1]
                    self.flops_tracker.update(batch_size, seq_len)

                    # Pre-compute conditions for this step
                    should_log = self.global_step % self.config.log_steps == 0
                    should_eval = (
                        self.config.eval_steps > 0
                        and self.global_step % self.config.eval_steps == 0
                    )
                    should_checkpoint = self.checkpoint_manager.should_save(self.global_step)

                    # Cache eval results to avoid running eval twice
                    all_eval_metrics: dict[str, dict[str, float]] | None = None

                    # Logging
                    if should_log:
                        log_metrics = self.metrics.compute_all()
                        log_metrics["learning_rate"] = get_lr(self.optimizer)
                        log_metrics["epoch"] = self.epoch
                        log_metrics["step"] = self.global_step

                        # Add masking frequency metrics
                        masking_freq = self.masking_frequency_tracker.compute()
                        for key, value in masking_freq.items():
                            log_metrics[f"train/masking_frequency/{key}"] = value

                        # Add FLOPs metrics
                        flops_metrics = self.flops_tracker.compute()
                        for key, value in flops_metrics.items():
                            log_metrics[f"train/{key}"] = value

                        if self.logger is not None:
                            # Use commit=False if eval will also log at this step
                            # to avoid wandb non-monotonic step warnings
                            self.logger.log(
                                log_metrics,
                                step=self.global_step,
                                commit=not should_eval,
                            )

                        self.metrics.reset()
                        self.masking_frequency_tracker.reset()

                    # Evaluation
                    if should_eval:
                        all_eval_metrics = self.evaluate_all()
                        if self.logger is not None and all_eval_metrics:
                            # Use log_eval_all if available, otherwise flatten and log
                            if hasattr(self.logger, "log_eval_all"):
                                self.logger.log_eval_all(all_eval_metrics, step=self.global_step)
                            else:
                                # Flatten metrics for basic logging
                                flat_metrics = {}
                                for eval_name, metrics in all_eval_metrics.items():
                                    for metric_name, value in metrics.items():
                                        flat_metrics[f"{eval_name}/{metric_name}"] = value
                                self.logger.log(flat_metrics, step=self.global_step)

                    # Checkpointing - reuse eval results if already computed
                    # Run eval on all ranks if needed (distributed dataloaders require it)
                    if should_checkpoint and all_eval_metrics is None and self.eval_dataloaders:
                        all_eval_metrics = self.evaluate_all()

                    # Only save checkpoints on main process to avoid file conflicts
                    if should_checkpoint and self.accelerator.is_main_process:
                        # Flatten for checkpoint manager
                        if all_eval_metrics:
                            eval_metrics = {}
                            for eval_name, metrics in all_eval_metrics.items():
                                for metric_name, value in metrics.items():
                                    eval_metrics[f"{eval_name}/{metric_name}"] = value
                        else:
                            eval_metrics = {}

                        self.checkpoint_manager.save(
                            step=self.global_step, epoch=self.epoch, metrics=eval_metrics
                        )

                    # Barrier after checkpointing to prevent other ranks from racing ahead
                    # while rank 0 saves (which can take minutes for large models)
                    if should_checkpoint:
                        self.accelerator.wait_for_everyone()

                    if self.global_step >= total_steps:
                        break

        progress_bar.close()

        # Final evaluation - all processes must participate (distributed dataloaders require it)
        if self.eval_dataloaders:
            all_eval_metrics = self.evaluate_all()
            final_metrics = {}
            for eval_name, metrics in all_eval_metrics.items():
                for metric_name, value in metrics.items():
                    final_metrics[f"{eval_name}/{metric_name}"] = value
        else:
            final_metrics = {}

        # Final checkpoint - only main process saves
        if self.accelerator.is_main_process:
            self.checkpoint_manager.save(
                step=self.global_step, epoch=self.epoch, metrics=final_metrics
            )
            if self.logger is not None:
                self.logger.finish()
