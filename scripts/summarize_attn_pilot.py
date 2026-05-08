"""Summarize the chain-aware attention ablation pilot from W&B.

Pulls runs in the ``somatic-attn-optimization`` project, prints a final-step
summary, and writes a Markdown table that can be pasted into the running log.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb


# Variant name -> (variant kind, model d_model, model n_layers, n_heads,
# expected non-emb params).
VARIANTS: dict[str, dict[str, object]] = {
    "separate_chain_aware": {
        "label": "separate-QKV chain-aware",
        "params": 24_011_264,
    },
    "shared_chain_aware": {
        "label": "shared-QKV chain-aware",
        "params": 19_292_672,
    },
    "standard_small": {
        "label": "standard MHA (same size)",
        "params": 19_292_672,
    },
    "standard_param_matched": {
        "label": "standard MHA (param-matched, d=288)",
        "params": 23_916_096,
    },
}

METRICS = [
    "train/loss",
    "eval/loss",
    "eval/perplexity",
    "eval/masked_accuracy",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", default="somatic-attn-optimization", help="W&B project name."
    )
    parser.add_argument("--entity", default=None, help="Optional W&B entity.")
    parser.add_argument(
        "--out",
        default="docs/attn_pilot_summary.md",
        help="Markdown output path.",
    )
    args = parser.parse_args()

    api = wandb.Api()
    project = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = api.runs(project)

    rows: list[dict[str, object]] = []
    for run in runs:
        if run.name not in VARIANTS:
            continue
        history = run.history(samples=10000, pandas=False, keys=METRICS)
        last = {m: None for m in METRICS}
        for row in history:
            for m in METRICS:
                if m in row and row[m] is not None:
                    last[m] = row[m]
        rows.append(
            {
                "name": run.name,
                "label": VARIANTS[run.name]["label"],
                "params": VARIANTS[run.name]["params"],
                "summary": dict(run.summary),
                "last": last,
                "runtime_s": run.summary.get("_runtime"),
            }
        )

    rows.sort(key=lambda r: list(VARIANTS).index(r["name"]))

    headers = [
        "variant",
        "non-emb params",
        "final train loss",
        "final eval loss",
        "final eval ppl",
        "final eval acc",
        "runtime (min)",
    ]
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        last = row["last"]
        runtime = (row["runtime_s"] or 0) / 60.0
        md_lines.append(
            "| {label} | {params:,} | {tl} | {el} | {ep} | {ea} | {rt:.1f} |".format(
                label=row["label"],
                params=row["params"],
                tl=_fmt(last.get("train/loss")),
                el=_fmt(last.get("eval/loss")),
                ep=_fmt(last.get("eval/perplexity")),
                ea=_fmt(last.get("eval/masked_accuracy")),
                rt=runtime,
            )
        )

    table = "\n".join(md_lines)
    print(table)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(table + "\n")
    print(f"\nWrote summary to {args.out}")


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value > 100:
            return f"{value:.2f}"
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
