"""Command-line interface for Somatic."""

from __future__ import annotations

from pathlib import Path

import click

from .version import __version__


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Somatic: Antibody Language Model."""
    pass


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(),
    default="configs",
    help="Config file (.yaml) or config directory (default: configs)",
)
@click.option(
    "--output-dir", "-o", type=click.Path(), default="outputs", help="Output directory"
)
@click.option("--name", "-n", default="somatic_experiment", help="Experiment name")
@click.option(
    "--resume", type=click.Path(exists=True), help="Checkpoint to resume from"
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--wandb/--no-wandb", default=True, help="Enable/disable WandB")
@click.argument("overrides", nargs=-1)
def train(
    config: str,
    output_dir: str,
    name: str,
    resume: str | None,
    seed: int,
    wandb: bool,
    overrides: tuple[str, ...],
) -> None:
    """Train a Somatic model.

    Training data must be specified in the config file (data.train) or via
    command-line override (data.train=/path/to/data.csv).

    Examples:

        somatic train data.train=/path/to/train.csv

        somatic train -c my_config.yaml data.train=/path/to/train.csv

        somatic train -c /path/to/configs data.train=/path/to/train.csv model=small
    """
    from .train import run_training

    run_training(
        config=config,
        output_dir=output_dir,
        name=name,
        resume_from=resume,
        seed=seed,
        use_wandb=wandb,
        overrides=list(overrides),
    )


@main.command()
@click.option(
    "--checkpoint",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Model checkpoint",
)
@click.option(
    "--input", "-i", type=click.Path(exists=True), required=True, help="Input file"
)
@click.option(
    "--output", "-o", type=click.Path(), required=True, help="Output file (.pt or .npy)"
)
@click.option(
    "--pooling",
    "-p",
    type=click.Choice(["mean", "cls", "max", "mean_max", "none"]),
    default="none",
    help="Pooling strategy",
)
@click.option("--batch-size", "-b", type=int, default=32, help="Batch size")
@click.option("--device", "-d", default=None, help="Device to run on")
def encode(
    checkpoint: str,
    input: str,
    output: str,
    pooling: str,
    batch_size: int,
    device: str | None,
) -> None:
    """Encode antibody sequences using a trained model.

    Examples:

        somatic encode -c checkpoints/best.pt -i data/seqs.csv -o embeddings.pt

        somatic encode -c model.pt -i seqs.parquet -o emb.npy --pooling mean
    """
    import numpy as np
    import pandas as pd
    import torch

    from .encoding import SomaticEncoder

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    click.echo(f"Loading model from {checkpoint}...")

    pooling_strategy = None if pooling == "none" else pooling
    encoder = SomaticEncoder.from_pretrained(checkpoint, device=device, pooling=pooling_strategy)

    click.echo(f"Loading data from {input}...")

    if input.endswith(".parquet"):
        df = pd.read_parquet(input)
    else:
        df = pd.read_csv(input)

    heavy_chains = df["heavy_chain"].tolist()
    light_chains = df["light_chain"].tolist()

    click.echo(f"Encoding {len(heavy_chains)} sequences...")

    return_numpy = output.endswith(".npy")
    embeddings = encoder.encode_batch(
        heavy_chains, light_chains, return_numpy=return_numpy, batch_size=batch_size
    )

    click.echo(f"Saving embeddings to {output}...")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output.endswith(".npy"):
        if isinstance(embeddings, list):
            np.save(output, np.array(embeddings, dtype=object))
        else:
            np.save(output, embeddings)
    elif output.endswith(".pt"):
        torch.save(embeddings, output)
    else:
        raise click.ClickException(f"Unknown output format: {output}")

    click.echo("Done!")


@main.command("model-size")
@click.option(
    "--config",
    "-c",
    type=click.Path(),
    default="configs",
    help="Config file (.yaml) or config directory (default: configs)",
)
@click.argument("overrides", nargs=-1)
def model_size(
    config: str,
    overrides: tuple[str, ...],
) -> None:
    """Get the number of trainable parameters for a model configuration.

    Examples:

        somatic model-size

        somatic model-size model=small

        somatic model-size model=large model.n_layers=32
    """
    import importlib.resources
    from contextlib import ExitStack

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from .model import SomaticConfig, SomaticModel

    with ExitStack() as stack:
        # Handle default/bundled configs
        if config is None or config == "configs":
            local_configs = Path("configs").absolute()
            if local_configs.exists() and local_configs.is_dir():
                config_dir = local_configs
            else:
                config_dir = stack.enter_context(
                    importlib.resources.as_file(
                        importlib.resources.files("somatic").joinpath("configs")
                    )
                )
            config_name = "config"
        else:
            config_path = Path(config).absolute()

            if not config_path.exists():
                raise click.ClickException(
                    f"Config path '{config}' does not exist.\n"
                    f"Provide a config file (.yaml) or config directory via --config/-c"
                )

            if config_path.is_file():
                config_dir = config_path.parent
                config_name = config_path.stem
            else:
                config_dir = config_path
                config_name = "config"

        stack.enter_context(
            initialize_config_dir(config_dir=str(config_dir), version_base=None)
        )

        cfg = compose(config_name=config_name, overrides=list(overrides))

    # Create model config
    model_config = SomaticConfig(
        vocab_size=cfg.model.vocab_size,
        padding_idx=cfg.model.padding_idx,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ffn=cfg.model.d_ffn,
        ffn_multiplier=cfg.model.ffn_multiplier,
        max_seq_len=cfg.model.max_seq_len,
        max_timesteps=cfg.model.max_timesteps,
        use_timestep_embedding=cfg.model.use_timestep_embedding,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        embedding_dropout=cfg.model.embedding_dropout,
        use_chain_aware_attention=cfg.model.use_chain_aware_attention,
    )

    # Create model and get parameter count
    model = SomaticModel(model_config)
    num_params = model.get_num_params()

    # Print results
    click.echo(f"Model configuration: {OmegaConf.to_yaml(cfg.model)}")
    click.echo(f"Trainable parameters: {num_params:,}")


if __name__ == "__main__":
    main()
