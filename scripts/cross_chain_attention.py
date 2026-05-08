"""Cross-chain attention analysis for trained Somatic models.

Loads a trained checkpoint, runs a fixed sample of eval sequences through it
with ``output_attentions=True``, then computes:

- per-layer mean cross-chain attention fraction;
- per-region (heavy/light × FWR/CDR1/CDR2/CDR3/FWR4) mean cross-chain
  attention fraction (region of the *query* position);
- per-position cross-chain attention density in heavy- and light-chain
  reference frames (shared x-axis);
- a per-position attention map showing where each query position attends
  across the concatenated [CLS] H L [EOS] sequence (used to look at the
  H→L boundary artifact).

For each variant the script writes a JSON summary plus a multi-panel PNG.
A combined comparison plot across variants is written separately.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from somatic.data.collator import AntibodyCollator
from somatic.data.dataset import AntibodyDataset
from somatic.model import SomaticModel


REGIONS_PER_CHAIN = ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]
REGION_LABELS = (
    ["CLS"]
    + [f"H_{r}" for r in REGIONS_PER_CHAIN]
    + [f"L_{r}" for r in REGIONS_PER_CHAIN]
    + ["EOS"]
)


def assign_regions(
    cdr_mask: torch.Tensor, chain_ids: torch.Tensor, special_tokens_mask: torch.Tensor
) -> torch.Tensor:
    """Assign a region id per token.

    The cdr_mask uses 0=FWR, 1=CDR1, 2=CDR2, 3=CDR3 within a chain. We expand
    that into FWR1/2/3/4 based on whether we're before/between/after CDRs of
    each chain.

    Returns
    -------
    Tensor of shape (B, S) with values in [0, len(REGION_LABELS)):
        0 = CLS, 1..7 = H_FWR1..H_FWR4 (and CDR1..CDR3), 8..14 = L_FWR1..L_FWR4
        15 = EOS, padding -> -1
    """
    B, S = cdr_mask.shape
    out = torch.full((B, S), -1, dtype=torch.long)

    for b in range(B):
        chain = chain_ids[b].tolist()
        cdr = cdr_mask[b].tolist()
        special = special_tokens_mask[b].tolist()

        # Heavy chain: state 0 = pre-CDR1, 1 = post-CDR1, 2 = post-CDR2, 3 = post-CDR3
        for chain_idx, base in [(0, 1), (1, 8)]:
            state = 0
            for i in range(S):
                if chain[i] != chain_idx:
                    continue
                if special[i]:
                    continue
                c = cdr[i]
                if c == 0:
                    region = ["FWR1", "FWR2", "FWR3", "FWR4"][state]
                else:
                    region = ["CDR1", "CDR2", "CDR3"][c - 1]
                    state = c  # advance after-CDR state when we see a CDR
                out[b, i] = base + REGIONS_PER_CHAIN.index(region)

        # specials
        for i in range(S):
            if special[i]:
                out[b, i] = 0 if i == 0 else (len(REGION_LABELS) - 1)

    return out


def compute_metrics(
    attentions: list[torch.Tensor],
    chain_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    region_ids: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    max_seq_len: int | None = None,
) -> dict:
    """Compute cross-chain metrics from a batch.

    Parameters
    ----------
    attentions
        List of (B, H, S, S) tensors, one per layer.
    chain_ids
        (B, S) chain identity (0 = heavy/CLS, 1 = light/EOS).
    attention_mask
        (B, S) 1 = valid token, 0 = padding.
    region_ids
        (B, S) region label ids (-1 = padding).
    special_tokens_mask
        (B, S) 1 = CLS/EOS, 0 = amino acid.
    """
    n_layers = len(attentions)
    B, _, S, _ = attentions[0].shape
    target_S = max_seq_len if max_seq_len is not None else S

    # build masks once
    attn_mask_b1s = attention_mask.unsqueeze(1)  # (B,1,S)
    valid_query = (attention_mask.bool() & ~special_tokens_mask)  # (B, S)
    cross_pair_mask = (chain_ids.unsqueeze(2) != chain_ids.unsqueeze(1))  # (B, S_q, S_k)
    valid_key = attention_mask.bool().unsqueeze(1)  # (B,1,S)
    cross_mask = cross_pair_mask & valid_key  # (B,S_q,S_k); valid keys only

    layer_cross = np.zeros(n_layers, dtype=np.float64)

    region_cross_sum = np.zeros((n_layers, len(REGION_LABELS)), dtype=np.float64)
    region_count = np.zeros((n_layers, len(REGION_LABELS)), dtype=np.float64)

    # Per-position cross-chain density. For the boundary artifact we want
    # to look at every concrete position in the concatenated sequence,
    # because the artifact is positional.
    # We aggregate (per layer, per absolute position) the cross-chain mass
    # for query positions that are amino-acid (not special, not pad).
    pos_cross_sum = np.zeros((n_layers, target_S), dtype=np.float64)
    pos_count = np.zeros((n_layers, target_S), dtype=np.float64)

    # Heatmap aggregation: average attention[i, j] across batch and heads.
    # Captured only for the *last* layer to keep memory reasonable.
    heatmap_last = torch.zeros((target_S, target_S), dtype=torch.float64)
    heatmap_count = torch.zeros((target_S, target_S), dtype=torch.float64)

    for L, attn in enumerate(attentions):
        # attn: (B, H, S, S). nan->0 already done in attention modules.
        # Cross-chain mass per (B, H, S_q): sum over S_k of attn * cross_mask
        cross_b1ss = cross_mask.unsqueeze(1).to(attn.dtype)  # (B,1,S_q,S_k)
        cross_per_query = (attn * cross_b1ss).sum(dim=-1)  # (B, H, S_q)

        # Mean over heads -> (B, S_q)
        cross_per_query = cross_per_query.mean(dim=1)

        # Apply valid-query mask
        valid_q = valid_query.float()
        layer_cross[L] = float(
            (cross_per_query * valid_q).sum().item() / max(valid_q.sum().item(), 1)
        )

        # Per-region aggregation: bin by region of query position
        rid = region_ids.clone()
        rid_flat = rid.flatten()
        cpq_flat = cross_per_query.flatten().to(torch.float64)
        valid_flat = (valid_q * (rid >= 0).float()).flatten().to(torch.bool)

        for r in range(len(REGION_LABELS)):
            sel = valid_flat & (rid_flat == r)
            if sel.any():
                region_cross_sum[L, r] += cpq_flat[sel].sum().item()
                region_count[L, r] += sel.sum().item()

        # Per-position aggregation (absolute position in seq).
        # cross_per_query shape (B, S_q). Sum over batch where valid.
        cpq_b_s = cross_per_query.to(torch.float64) * valid_q.to(torch.float64)
        pos_cross_sum[L, :S] += cpq_b_s.sum(dim=0).cpu().numpy()
        pos_count[L, :S] += valid_q.sum(dim=0).cpu().numpy()

        if L == n_layers - 1:
            # Heatmap (head-mean, batch-mean conditioned on validity at both ends).
            attn_hm = attn.mean(dim=1)  # (B, S, S)
            mask_qk = valid_query.unsqueeze(2).float() * valid_key.float()  # (B,S,S)
            heatmap_last[:S, :S] += (
                attn_hm.to(torch.float64) * mask_qk.to(torch.float64)
            ).sum(dim=0).cpu()
            heatmap_count[:S, :S] += mask_qk.to(torch.float64).sum(dim=0).cpu()

    # Normalize
    region_mean = np.zeros_like(region_cross_sum)
    region_mean = np.where(region_count > 0, region_cross_sum / np.maximum(region_count, 1), 0.0)
    pos_mean = np.where(pos_count > 0, pos_cross_sum / np.maximum(pos_count, 1), 0.0)
    heatmap_mean = (heatmap_last / heatmap_count.clamp(min=1)).numpy()

    return {
        "layer_cross": layer_cross,
        "region_mean": region_mean,
        "region_count": region_count,
        "pos_mean": pos_mean,
        "pos_count": pos_count,
        "heatmap_last": heatmap_mean,
    }


def aggregate(metrics_list: list[dict]) -> dict:
    """Aggregate metrics across batches by weighted mean."""
    n = len(metrics_list)
    n_layers = metrics_list[0]["layer_cross"].shape[0]
    S = metrics_list[0]["pos_mean"].shape[1]
    n_regions = metrics_list[0]["region_mean"].shape[1]

    layer_cross = np.zeros(n_layers)
    region_sum = np.zeros((n_layers, n_regions))
    region_count = np.zeros((n_layers, n_regions))
    pos_sum = np.zeros((n_layers, S))
    pos_count = np.zeros((n_layers, S))
    heatmap_sum = np.zeros((S, S))
    weight_total = 0.0

    for m in metrics_list:
        # naive averaging across batches; each batch contributed equal weight.
        layer_cross += m["layer_cross"] / n
        region_sum += m["region_mean"] * m["region_count"]
        region_count += m["region_count"]
        pos_sum += m["pos_mean"] * m["pos_count"]
        pos_count += m["pos_count"]
        heatmap_sum += m["heatmap_last"]  # already mean per batch
        weight_total += 1

    region_mean = np.where(region_count > 0, region_sum / np.maximum(region_count, 1), 0.0)
    pos_mean = np.where(pos_count > 0, pos_sum / np.maximum(pos_count, 1), 0.0)
    heatmap_mean = heatmap_sum / max(weight_total, 1)
    return {
        "layer_cross": layer_cross,
        "region_mean": region_mean,
        "region_count": region_count,
        "pos_mean": pos_mean,
        "pos_count": pos_count,
        "heatmap_last": heatmap_mean,
    }


def plot_variant(name: str, agg: dict, out_dir: Path) -> None:
    n_layers, S = agg["pos_mean"].shape

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Per-layer cross-chain fraction
    axes[0, 0].plot(np.arange(n_layers), agg["layer_cross"], "-o")
    axes[0, 0].set_title(f"{name}: cross-chain attention fraction by layer")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Mean cross-chain attention mass")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Per-region (averaged across layers; bar chart)
    region_overall = agg["region_mean"].mean(axis=0)
    region_idx = np.arange(len(REGION_LABELS))
    colors = ["#888"] + ["#1f77b4"] * 7 + ["#d62728"] * 7 + ["#888"]
    axes[0, 1].bar(region_idx, region_overall, color=colors)
    axes[0, 1].set_title(f"{name}: cross-chain attention by query region (layer-mean)")
    axes[0, 1].set_xticks(region_idx)
    axes[0, 1].set_xticklabels(REGION_LABELS, rotation=60, ha="right")
    axes[0, 1].set_ylabel("Mean cross-chain attention mass")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # 3. Per-position cross-chain attention (layer-mean) over absolute position
    pos_overall = agg["pos_mean"].mean(axis=0)
    valid_pos = agg["pos_count"].sum(axis=0) > 0
    x = np.arange(S)
    axes[1, 0].plot(x[valid_pos], pos_overall[valid_pos], "-")
    axes[1, 0].set_title(f"{name}: cross-chain attention by absolute position (layer-mean)")
    axes[1, 0].set_xlabel("Absolute position in [CLS] H L [EOS]")
    axes[1, 0].set_ylabel("Mean cross-chain attention mass")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Last-layer mean attention heatmap
    hm = agg["heatmap_last"]
    im = axes[1, 1].imshow(hm, cmap="viridis", aspect="auto", origin="upper")
    axes[1, 1].set_title(f"{name}: mean attention map (last layer, batch+head mean)")
    axes[1, 1].set_xlabel("Key position")
    axes[1, 1].set_ylabel("Query position")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)

    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=140)
    plt.close(fig)


def find_chain_boundary(pos_count: np.ndarray) -> tuple[int, int]:
    """Estimate the H→L boundary range from positional coverage.

    We use the median sample's last heavy position and first light position by
    relying on the cdr_mask coverage (which only fires for amino-acid tokens).
    This is approximate and is only used to set a x-zoom for plotting.
    """
    valid = (pos_count.sum(axis=0) > 0)
    nonzero = np.where(valid)[0]
    if len(nonzero) == 0:
        return 0, 0
    return int(nonzero[0]), int(nonzero[-1])


def plot_comparison(by_variant: dict[str, dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Layer-wise overlay
    for name, agg in by_variant.items():
        axes[0].plot(np.arange(len(agg["layer_cross"])), agg["layer_cross"], "-o", label=name)
    axes[0].set_title("Cross-chain attention fraction by layer (per variant)")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean cross-chain attention mass")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Per-position overlay
    for name, agg in by_variant.items():
        pos_overall = agg["pos_mean"].mean(axis=0)
        valid_pos = agg["pos_count"].sum(axis=0) > 0
        x = np.arange(pos_overall.shape[0])
        axes[1].plot(x[valid_pos], pos_overall[valid_pos], "-", label=name)
    axes[1].set_title("Cross-chain attention by absolute position (layer-mean) per variant")
    axes[1].set_xlabel("Absolute position in [CLS] H L [EOS]")
    axes[1].set_ylabel("Mean cross-chain attention mass")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "_comparison.png", dpi=140)
    plt.close(fig)

    # Region overlay: average across layers, group by chain.
    fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    width = 0.15
    x = np.arange(len(REGION_LABELS))
    for i, (name, agg) in enumerate(by_variant.items()):
        region_overall = agg["region_mean"].mean(axis=0)
        axes.bar(x + i * width, region_overall, width=width, label=name)
    axes.set_xticks(x + width * (len(by_variant) - 1) / 2)
    axes.set_xticklabels(REGION_LABELS, rotation=60, ha="right")
    axes.set_title("Cross-chain attention by query region — variants overlaid")
    axes.set_ylabel("Mean cross-chain attention mass (layer-mean)")
    axes.legend()
    axes.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "_comparison_regions.png", dpi=140)
    plt.close(fig)


def run_variant(
    name: str,
    checkpoint: Path,
    eval_path: Path,
    n_sequences: int,
    batch_size: int,
    device: str,
    out_dir: Path,
) -> dict:
    print(f"\n=== {name} ===")
    print(f"  loading: {checkpoint}")

    model = SomaticModel.from_pretrained(str(checkpoint), map_location=device)
    model.eval()
    model.to(device)
    cfg = model.config
    print(f"  config: d_model={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} mode={cfg.chain_aware_projection_mode} chain_aware={cfg.use_chain_aware_attention}")

    dataset = AntibodyDataset(
        data_path=str(eval_path),
        max_length=cfg.max_seq_len,
        heavy_col="sequence_aa:0",
        light_col="sequence_aa:1",
        heavy_cdr_col="cdr_mask_aa:0",
        light_cdr_col="cdr_mask_aa:1",
        heavy_nongermline_col="nongermline_mask_aa:0",
        light_nongermline_col="nongermline_mask_aa:1",
    )
    # subsample to n_sequences
    if n_sequences < len(dataset):
        dataset.df = dataset.df.iloc[:n_sequences].reset_index(drop=True)

    collator = AntibodyCollator(max_length=cfg.max_seq_len, pad_to_max=False)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=2
    )

    metrics_per_batch = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(
                token_ids=batch["token_ids"],
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
                output_attentions=True,
            )
            attentions = outputs["attentions"]
            region_ids = assign_regions(
                cdr_mask=batch["cdr_mask"].cpu(),
                chain_ids=batch["chain_ids"].cpu(),
                special_tokens_mask=batch["special_tokens_mask"].cpu(),
            ).to(device)
            m = compute_metrics(
                attentions=attentions,
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
                region_ids=region_ids,
                special_tokens_mask=batch["special_tokens_mask"],
                max_seq_len=cfg.max_seq_len,
            )
            metrics_per_batch.append(m)

    agg = aggregate(metrics_per_batch)
    print(f"  layers cross-chain mean: {agg['layer_cross'].mean():.4f}  (range {agg['layer_cross'].min():.4f} – {agg['layer_cross'].max():.4f})")

    # save JSON
    out = {
        "name": name,
        "config": {
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "max_seq_len": cfg.max_seq_len,
            "use_chain_aware_attention": cfg.use_chain_aware_attention,
            "chain_aware_projection_mode": cfg.chain_aware_projection_mode,
        },
        "layer_cross": agg["layer_cross"].tolist(),
        "region_labels": REGION_LABELS,
        "region_mean_per_layer": agg["region_mean"].tolist(),
        "region_count_per_layer": agg["region_count"].tolist(),
        "pos_mean_per_layer": agg["pos_mean"].tolist(),
        "pos_count_per_layer": agg["pos_count"].tolist(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w") as fh:
        json.dump(out, fh)

    plot_variant(name, agg, out_dir)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        default="/home/jovyan/work/somatic/somatic/outputs/attn_ablation_v2",
        help="Root containing one subdirectory per variant.",
    )
    parser.add_argument(
        "--eval-path",
        default="/home/jovyan/work/ablm_training-data/v2026-05-03/train-test-eval_splits/train-clust95_testeval-clust80/minimal/L2677-eval_unclustered.parquet",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            "separate_chain_aware",
            "shared_chain_aware",
            "shared_chain_aware_pm",
            "standard_small",
            "standard_param_matched",
        ],
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="checkpoints/best_checkpoint.pt",
        help="Path within each variant directory to the checkpoint to load.",
    )
    parser.add_argument("--n-sequences", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--out-dir",
        default="/home/jovyan/work/somatic/somatic/outputs/attn_ablation_v2/_analysis",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_variant: dict[str, dict] = {}
    for name in args.variants:
        ckpt_dir = Path(args.checkpoint_root) / name
        ckpt = ckpt_dir / args.checkpoint_glob
        if not ckpt.exists():
            # fall back: pick the largest-step periodic checkpoint
            cps = sorted(
                ckpt_dir.glob("checkpoints/checkpoint_step_*.pt"),
                key=lambda p: int(p.stem.split("_")[-1]),
            )
            if not cps:
                print(f"[skip] no checkpoint at {ckpt}")
                continue
            ckpt = cps[-1]
        agg = run_variant(
            name=name,
            checkpoint=ckpt,
            eval_path=Path(args.eval_path),
            n_sequences=args.n_sequences,
            batch_size=args.batch_size,
            device=device,
            out_dir=out_dir,
        )
        by_variant[name] = agg

    if by_variant:
        plot_comparison(by_variant, out_dir)

    # Tabular cross-variant summary
    rows = []
    for name, agg in by_variant.items():
        rows.append(
            {
                "name": name,
                "mean_cross": float(agg["layer_cross"].mean()),
                "min_cross": float(agg["layer_cross"].min()),
                "max_cross": float(agg["layer_cross"].max()),
            }
        )
    df = pd.DataFrame(rows)
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    df.to_csv(out_dir / "_summary.csv", index=False)


if __name__ == "__main__":
    main()
