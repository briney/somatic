"""Main training loop with Accelerate integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from accelerate import Accelerator

from ..masking import InformationWeightedMasker, UniformMasker
from ..utils.progress import ProgressManager
from .checkpoint import CheckpointConfig, CheckpointManager
from .flops import FLOPsConfig, FLOPsTracker
from .masking_frequency import MaskingFrequencyConfig, MaskingFrequencyTracker
from .metrics import (
    MetricAccumulator,
    MLMMetrics,
    compute_mlm_metrics,
)
from .optimizer import create_optimizer, create_scheduler, get_lr

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from ..eval import Evaluator
    from ..model import SomaticForMaskedLM


_VALID_COMPILE_MODES = {"default", "reduce-overhead", "max-autotune"}


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

    # Mixed precision: "auto" | "no" | "fp16" | "bf16" | "fp8"
    # "auto" defers to accelerate's own resolution (env / config file / launch flag).
    mixed_precision: str = "auto"

    # torch.compile
    # Compiles the model with torch.compile AFTER accelerate.prepare (so accelerate
    # wraps the raw model with DDP first; see Trainer.__init__ for the ordering
    # rationale). Uses dynamic=True so variable per-batch sequence lengths do not
    # trigger recompilation. Disabled by default.
    compile: bool = False
    # Compilation mode passed to torch.compile(mode=...).
    # "default"          — balanced; safe for dynamic-shape training.
    # "reduce-overhead"  — CUDA graphs; for fixed-shape inference only (conflicts
    #                      with dynamic shapes + DDP).
    # "max-autotune"     — longest compile, best peak throughput (e.g. on Blackwell).
    compile_mode: str = "default"

    def __post_init__(self) -> None:
        """Validate training configuration."""
        if self.compile_mode not in _VALID_COMPILE_MODES:
            raise ValueError(
                f"compile_mode must be one of {_VALID_COMPILE_MODES}, got '{self.compile_mode}'"
            )


class Trainer:
    """Main trainer class with Accelerate integration."""

    def __init__(
        self,
        config: TrainingConfig,
        model: SomaticForMaskedLM,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        eval_dataloaders: dict[str, DataLoader] | None = None,
        evaluator: Evaluator | None = None,
        accelerator: Accelerator | None = None,
        masking_frequency_config: MaskingFrequencyConfig | None = None,
        flops_config: FLOPsConfig | None = None,
    ) -> None:
        self.config = config

        # Use provided accelerator or create a new one
        if accelerator is not None:
            self.accelerator = accelerator
        else:
            # "auto" defers to accelerate's own resolution; any other value is authoritative.
            accel_kwargs: dict = {
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
            }
            if config.mixed_precision != "auto":
                accel_kwargs["mixed_precision"] = config.mixed_precision
            self.accelerator = Accelerator(**accel_kwargs)

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

        # Apply torch.compile AFTER accelerator.prepare so accelerate always wraps the
        # raw model with DDP first. Compiling the DDP-wrapped model gives
        # OptimizedModule(_orig_mod=DDP(model)), which accelerate.unwrap_model peels
        # unambiguously. Compiling *before* prepare causes some accelerate versions to
        # strip the compile wrapper during prepare, leaving DDP(model) with no
        # _orig_mod — and unwrap_model then fails with KeyError: '_orig_mod' at the
        # first eval/checkpoint. (Eval runs on the raw, uncompiled `model` reference.)
        if config.compile:
            # Selective activation checkpointing is incompatible with the default
            # DDPOptimizer: it splits the compiled graph at gradient-bucket boundaries
            # to overlap allreduce with backward, which fragments each block's AC
            # higher-order op across subgraphs. AOT autograd's min-cut partitioner then
            # can't honor the SAC MUST_SAVE set, so `selective` silently collapses to
            # full recompute under DDP+compile (single-GPU is unaffected — DDPOptimizer
            # never engages). Disabling graph-splitting keeps SAC intact; the only cost
            # is the lost comm/compute overlap. Gate on `selective` only so `full`/`none`
            # keep the default overlap. The flag is a process-global dynamo config.
            if getattr(model.config, "gradient_checkpointing_mode", "none") == "selective":
                # `from torch import _dynamo` (not `import torch._dynamo`) so the local
                # binding is `_dynamo`, not `torch` — the latter would shadow the
                # module-level `torch` used elsewhere in this method.
                from torch import _dynamo

                _dynamo.config.optimize_ddp = False
            self.model = torch.compile(self.model, dynamic=True, mode=config.compile_mode)

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
        # keep_torch_compile=False peels both the DDP and torch.compile wrappers,
        # yielding the raw SomaticModel — clean (no `_orig_mod.` prefix) state_dict
        # keys for checkpointing and direct `.config` access. Harmless when not
        # compiled. Without this, a compiled model would save unloadable checkpoints.
        unwrapped_model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_config,
            unwrapped_model,
            self.optimizer,
            self.scheduler,
        )

        self.metrics = MetricAccumulator()
        self.global_step = 0
        self.epoch = 0.0
        self.steps_per_epoch = len(self.train_dataloader)
        self.logger = None
        self._progress_manager: ProgressManager | None = None

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
            model_config=unwrapped_model.config,
            world_size=self.accelerator.num_processes,
        )

    def set_logger(self, logger) -> None:
        """Set the logger for training metrics."""
        self.logger = logger

    def set_evaluator(self, evaluator: Evaluator) -> None:
        """Set the evaluator for advanced metrics.

        Args:
            evaluator: Evaluator instance for computing metrics.
        """
        self.evaluator = evaluator

    def _apply_masking(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply masking to a batch.

        Returns a dict with ``masked_input_ids`` (model input) and ``labels``
        (original ids at masked positions, ``-100`` elsewhere).
        """
        if self.config.use_information_weighted_masking and (
            batch.get("cdr_mask") is not None or batch.get("non_templated_mask") is not None
        ):
            masked_input_ids, labels = self.masker.apply_mask(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                cdr_mask=batch.get("cdr_mask"),
                non_templated_mask=batch.get("non_templated_mask"),
                special_tokens_mask=batch.get("special_tokens_mask"),
            )
        else:
            masked_input_ids, labels = self.uniform_masker.apply_mask(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                special_tokens_mask=batch.get("special_tokens_mask"),
            )

        return {"masked_input_ids": masked_input_ids, "labels": labels}

    def training_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, MLMMetrics]:
        """Execute a single training step.

        Returns:
            Tuple of (loss tensor for backprop, MLMMetrics with all metrics).
        """
        mask_output = self._apply_masking(batch)
        labels = mask_output["labels"]
        mask = labels != -100

        # Track masking frequency
        self.masking_frequency_tracker.update(mask, batch)

        outputs = self.model(
            input_ids=mask_output["masked_input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )

        metrics = compute_mlm_metrics(
            logits=outputs.logits,
            targets=batch["input_ids"],
            mask_labels=mask,
            attention_mask=batch["attention_mask"],
        )

        # The model computes the masked cross-entropy internally from `labels`.
        return outputs.loss, metrics

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the eval dataset (legacy single-dataset method)."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        eval_metrics = MetricAccumulator()

        disable_progress = not self.accelerator.is_local_main_process
        with ProgressManager.standalone_eval_task(
            "Evaluating", total=len(self.eval_dataloader), disable=disable_progress
        ) as progress_task:
            for batch in self.eval_dataloader:
                mask_output = self._apply_masking(batch)
                labels = mask_output["labels"]

                outputs = self.model(
                    input_ids=mask_output["masked_input_ids"],
                    token_type_ids=batch["token_type_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels,
                )

                metrics = compute_mlm_metrics(
                    logits=outputs.logits,
                    targets=batch["input_ids"],
                    mask_labels=labels != -100,
                    attention_mask=batch["attention_mask"],
                )

                eval_metrics.update("loss", metrics.loss)
                eval_metrics.update("accuracy", metrics.accuracy)
                eval_metrics.update("perplexity", metrics.perplexity)
                progress_task.advance()

        self.model.train()

        # These metrics are populated every eval batch; coalesce guards the
        # empty-dataloader edge case where compute() returns None.
        return {
            "val_loss": eval_metrics.compute("loss") or 0.0,
            "val_accuracy": eval_metrics.compute("accuracy") or 0.0,
            "val_perplexity": eval_metrics.compute("perplexity") or 0.0,
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
                progress=self._progress_manager,
            )

        # Fall back to simple evaluation for each dataset
        all_results: dict[str, dict[str, float]] = {}
        self.model.eval()

        disable_progress = not self.accelerator.is_local_main_process

        for eval_name, eval_loader in self.eval_dataloaders.items():
            eval_metrics = MetricAccumulator()

            # Get or create masking frequency tracker for this eval dataset
            if eval_name not in self.eval_masking_frequency_trackers:
                self.eval_masking_frequency_trackers[eval_name] = MaskingFrequencyTracker(
                    self.masking_frequency_config
                )
            eval_tracker = self.eval_masking_frequency_trackers[eval_name]
            eval_tracker.reset()

            eval_task_cm = (
                self._progress_manager.eval_task(f"Eval ({eval_name})", total=len(eval_loader))
                if self._progress_manager is not None
                else ProgressManager.standalone_eval_task(
                    f"Eval ({eval_name})",
                    total=len(eval_loader),
                    disable=disable_progress,
                )
            )

            with eval_task_cm as progress_task:
                for batch in eval_loader:
                    mask_output = self._apply_masking(batch)
                    labels = mask_output["labels"]
                    mask = labels != -100

                    # Track masking frequency for eval
                    eval_tracker.update(mask, batch)

                    outputs = self.model(
                        input_ids=mask_output["masked_input_ids"],
                        token_type_ids=batch["token_type_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=labels,
                    )

                    metrics = compute_mlm_metrics(
                        logits=outputs.logits,
                        targets=batch["input_ids"],
                        mask_labels=mask,
                        attention_mask=batch["attention_mask"],
                    )

                    eval_metrics.update("loss", metrics.loss)
                    eval_metrics.update("accuracy", metrics.accuracy)
                    eval_metrics.update("perplexity", metrics.perplexity)
                    progress_task.advance()

            all_results[eval_name] = {
                "loss": eval_metrics.compute("loss") or 0.0,
                "accuracy": eval_metrics.compute("accuracy") or 0.0,
                "perplexity": eval_metrics.compute("perplexity") or 0.0,
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

        with ProgressManager(accelerator=self.accelerator) as progress:
            self._progress_manager = progress
            progress.train_task(total=total_steps, start=self.global_step)

            self._train_loop(total_steps, progress)

            # Final evaluation runs inside the live progress display so the
            # eval sub-bars render under the (now-complete) training bar.
            if self.eval_dataloaders:
                progress.start_eval_cycle()
                all_eval_metrics = self.evaluate_all()
                final_metrics = {}
                for eval_name, metrics in all_eval_metrics.items():
                    for metric_name, value in metrics.items():
                        final_metrics[f"{eval_name}/{metric_name}"] = value
            else:
                final_metrics = {}

        self._progress_manager = None

        # Final checkpoint - only main process saves
        if self.accelerator.is_main_process:
            self.checkpoint_manager.save(
                step=self.global_step, epoch=self.epoch, metrics=final_metrics
            )
            if self.logger is not None:
                self.logger.finish()

    def _train_loop(self, total_steps: int, progress: ProgressManager) -> None:
        """Inner training loop, scoped to the live progress display."""
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
                    progress.advance_train()

                    self.metrics.update("train/loss", step_metrics.loss)
                    self.metrics.update("train/accuracy", step_metrics.accuracy)
                    self.metrics.update("train/perplexity", step_metrics.perplexity)

                    # Update FLOPs tracking
                    batch_size = batch["input_ids"].shape[0]
                    seq_len = batch["input_ids"].shape[1]
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
                        progress.start_eval_cycle()
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
                        progress.start_eval_cycle()
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
