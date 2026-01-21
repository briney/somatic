"""Training entry point for use with accelerate launch."""

from __future__ import annotations

import importlib.resources
from contextlib import ExitStack
from pathlib import Path

import torch
from accelerate import Accelerator
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from .data import create_eval_dataloaders, create_train_dataloader
from .eval import Evaluator
from .logging import WandbLogger
from .model import SomaticConfig, SomaticModel
from .training import FLOPsConfig, MaskingFrequencyConfig, Trainer, TrainingConfig
from .utils import set_seed


def _load_config(
    config: str | None,
    name: str,
    seed: int,
    output_dir: str,
    overrides: list[str] | None,
) -> DictConfig:
    """Load configuration - runs identically on all processes.

    This function must be deterministic across all distributed processes
    to ensure they have identical configuration state before Accelerator
    initialization.

    Parameters
    ----------
    config
        Path to config file (.yaml/.yml) or config directory.
    name
        Experiment name.
    seed
        Random seed.
    output_dir
        Output directory.
    overrides
        List of Hydra config overrides.

    Returns
    -------
    DictConfig
        Loaded configuration.
    """
    with ExitStack() as stack:
        # Handle default/bundled configs
        if config is None or config == "configs":
            # Check if local configs directory exists first
            local_configs = Path("configs").absolute()
            if local_configs.exists() and local_configs.is_dir():
                config_dir = local_configs
            else:
                # Use bundled configs from package
                config_dir = stack.enter_context(
                    importlib.resources.as_file(
                        importlib.resources.files("somatic").joinpath("configs")
                    )
                )
            config_name = "config"
        else:
            config_path = Path(config).absolute()

            # Validate config path exists
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Config path '{config}' does not exist.\n"
                    f"Provide a config file (.yaml) or config directory via --config/-c"
                )

            # Determine if config is a file or directory
            if config_path.is_file():
                # Config file provided - use parent as config dir
                config_dir = config_path.parent
                config_name = config_path.stem
            else:
                # Config directory provided
                config_dir = config_path
                config_name = "config"

        stack.enter_context(initialize_config_dir(config_dir=str(config_dir), version_base=None))

        override_list = overrides or []
        override_list.extend([f"name={name}", f"seed={seed}"])
        # Only override output_dir if explicitly provided (not default)
        # This allows the config's output_dir: outputs/${name} interpolation to work
        if output_dir != "outputs":
            override_list.append(f"output_dir={output_dir}")

        return compose(config_name=config_name, overrides=override_list)


def _build_masking_frequency_config(cfg: DictConfig) -> MaskingFrequencyConfig:
    """Build MaskingFrequencyConfig from Hydra config.

    Parameters
    ----------
    cfg
        Full Hydra configuration.

    Returns
    -------
    MaskingFrequencyConfig
        Masking frequency configuration for the tracker.
    """
    mf_cfg = cfg.train.get("masking_frequency", {})
    if not mf_cfg:
        return MaskingFrequencyConfig()

    # Build config from available fields
    config_kwargs = {}
    for field_name in MaskingFrequencyConfig.__dataclass_fields__:
        if field_name in mf_cfg:
            config_kwargs[field_name] = mf_cfg[field_name]

    return MaskingFrequencyConfig(**config_kwargs)


def _build_flops_config(cfg: DictConfig) -> FLOPsConfig:
    """Build FLOPsConfig from Hydra config.

    Parameters
    ----------
    cfg
        Full Hydra configuration.

    Returns
    -------
    FLOPsConfig
        FLOPs tracking configuration.
    """
    flops_cfg = cfg.train.get("flops", {})
    return FLOPsConfig(enabled=flops_cfg.get("enabled", True))


def run_training(
    config: str | None = None,
    output_dir: str = "outputs",
    name: str = "somatic_experiment",
    resume_from: str | None = None,
    seed: int = 42,
    use_wandb: bool = True,
    overrides: list[str] | None = None,
) -> None:
    """Main training function.

    Parameters
    ----------
    config
        Path to config file (.yaml/.yml) or config directory. If a file is
        provided, its parent directory is used as the config directory.
        If None or "configs", uses bundled default configs.
    output_dir
        Output directory for checkpoints and logs.
    name
        Experiment name.
    resume_from
        Optional checkpoint path to resume from.
    seed
        Random seed.
    use_wandb
        Whether to enable WandB logging.
    overrides
        List of Hydra config overrides (including data.train for training data).
    """
    # ==================================================================
    # PHASE 1: Pre-distributed setup (ALL processes do this identically)
    # ==================================================================

    # 1. Load config FIRST (before any distributed operations)
    # All processes must have identical config before Accelerator init
    cfg = _load_config(config, name, seed, output_dir, overrides)

    # 2. Set seed BEFORE Accelerator (ensures RNG synchronization)
    set_seed(cfg.seed)

    # ==================================================================
    # PHASE 2: Distributed initialization
    # ==================================================================

    # 3. Create Accelerator (NOW all processes are in sync with same state)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
    )
    is_main = accelerator.is_main_process

    # 4. wandb login (early, but AFTER Accelerator - main process only)
    if use_wandb and is_main:
        try:
            import wandb

            # This will prompt for login if not already authenticated
            wandb.login()
        except ImportError:
            pass  # wandb not installed, will be handled later

    # ==================================================================
    # PHASE 3: Main-process-only I/O
    # ==================================================================

    accelerator.print(OmegaConf.to_yaml(cfg))
    if is_main:
        output_path = Path(cfg.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_path / "config.yaml")

    # 5. Synchronize AFTER all divergent operations
    accelerator.wait_for_everyone()

    # ==================================================================
    # PHASE 4: Parallel operations (all processes)
    # ==================================================================

    # Create model
    model_config = SomaticConfig(
        vocab_size=cfg.model.vocab_size,
        padding_idx=cfg.model.padding_idx,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ffn=cfg.model.d_ffn,
        ffn_multiplier=cfg.model.ffn_multiplier,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        embedding_dropout=cfg.model.embedding_dropout,
        use_chain_aware_attention=cfg.model.use_chain_aware_attention,
    )
    model = SomaticModel(model_config)
    accelerator.print(f"Model parameters: {model.get_num_params():,}")

    # Create train dataloader (handles single or multi-dataset automatically)
    train_loader = create_train_dataloader(
        cfg=cfg.data,
        batch_size=cfg.train.batch_size,
    )

    # Create eval dataloaders (handles multiple eval datasets)
    eval_dataloaders = create_eval_dataloaders(
        cfg=cfg.data,
        default_batch_size=cfg.train.batch_size,
    )

    # Create training config
    training_config = TrainingConfig(
        max_steps=cfg.train.max_steps,
        max_epochs=cfg.train.max_epochs,
        batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.optimizer.learning_rate,
        weight_decay=cfg.train.optimizer.weight_decay,
        betas=tuple(cfg.train.optimizer.betas),
        max_grad_norm=cfg.train.max_grad_norm,
        scheduler_decay=cfg.train.scheduler.decay,
        warmup_steps=cfg.train.scheduler.warmup_steps,
        min_lr_ratio=cfg.train.scheduler.min_lr_ratio,
        mask_rate=cfg.masking.mask_rate,
        cdr_weight_multiplier=cfg.masking.cdr_weight_multiplier,
        nongermline_weight_multiplier=cfg.masking.nongermline_weight_multiplier,
        use_information_weighted_masking=cfg.masking.use_information_weighted_masking,
        masking_selection=cfg.masking.masking_selection,
        log_steps=cfg.train.log_steps,
        eval_steps=cfg.train.eval_steps,
        checkpoint_steps=cfg.train.checkpoint_steps,
        checkpoint_dir=cfg.train.checkpoint_dir,
        keep_last_n_checkpoints=cfg.train.keep_last_n_checkpoints,
        save_best=cfg.train.save_best,
        seed=cfg.seed,
        mixed_precision=cfg.train.mixed_precision,
    )

    # Build masking frequency config
    masking_frequency_config = _build_masking_frequency_config(cfg)

    # Build FLOPs tracking config
    flops_config = _build_flops_config(cfg)

    # Create trainer with pre-created accelerator
    trainer = Trainer(
        config=training_config,
        model=model,
        train_dataloader=train_loader,
        eval_dataloaders=eval_dataloaders if eval_dataloaders else None,
        accelerator=accelerator,
        masking_frequency_config=masking_frequency_config,
        flops_config=flops_config,
    )

    # Create evaluator for advanced metrics (including region-based eval)
    evaluator = Evaluator(
        cfg=cfg,
        model=model,
        accelerator=accelerator,
    )
    trainer.set_evaluator(evaluator)

    # Warn if multi-GPU available but not being used
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = accelerator.num_processes
    if world_size == 1 and num_gpus > 1:
        accelerator.print(
            f"WARNING: {num_gpus} GPUs detected but only 1 process active.\n"
            f"For multi-GPU training, use: accelerate launch -m somatic.train ..."
        )
    elif world_size > 1:
        accelerator.print(f"Distributed training with {world_size} processes")

    # Set up logging (only on main process)
    if use_wandb and cfg.log.wandb.enabled and is_main:
        logger = WandbLogger(
            project=cfg.log.wandb.project,
            name=cfg.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            entity=cfg.log.wandb.entity,
            tags=list(cfg.log.wandb.tags) if cfg.log.wandb.tags else None,
            notes=cfg.log.wandb.notes,
        )
        trainer.set_logger(logger)

    # Resume if specified
    if resume_from:
        accelerator.print(f"Resuming from {resume_from}")
        trainer.checkpoint_manager.load(resume_from)

    # Train
    trainer.train()


def main() -> None:
    """Entry point for `accelerate launch -m somatic.train`.

    Training data must be specified via config overrides, e.g.:
        accelerate launch -m somatic.train data.train=/path/to/train.csv
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a Somatic model",
        epilog="Training data must be specified via override: data.train=/path/to/data.csv",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs",
        help="Config file (.yaml) or config directory (default: configs)",
    )
    parser.add_argument("--output-dir", "-o", default="outputs", help="Output directory")
    parser.add_argument("--name", "-n", default="somatic_experiment", help="Experiment name")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")

    args, unknown = parser.parse_known_args()

    run_training(
        config=args.config,
        output_dir=args.output_dir,
        name=args.name,
        resume_from=args.resume,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        overrides=unknown,
    )


if __name__ == "__main__":
    main()
