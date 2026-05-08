"""Summarize the V2 chain-aware attention ablation pilot from W&B.

Same pattern as ``summarize_attn_pilot.py`` but for the 5-variant V2 pilot
in the ``somatic-attn-optimization-v2`` project. Pulls the eval-step
trajectories every 5k for each variant and writes a Markdown table that
can be pasted into the V2 lab notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb


VARIANTS: dict[str, dict[str, object]] = {
    "separate_chain_aware": {
        "label": "separate-QKV chain-aware",
        "params": 24_011_264,
    },
    "shared_chain_aware": {
        "label": "shared-QKV chain-aware",
        "params": 19_292_672,
    },
    "shared_chain_aware_pm": {
        "label": "shared-QKV chain-aware (param-matched)",
        "params": 23_916_096,
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
    "eval/ppl",
    "eval/mask_acc",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="somatic-attn-optimization-v2")
    parser.add_argument("--entity", default="thebrineylab")
    parser.add_argument(
        "--out",
        default="docs/attn_pilot_v2_summary.md",
    )
    args = parser.parse_args()

    api = wandb.Api()
    project = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = api.runs(project)
    runs_by_name = {r.name: r for r in runs if r.name in VARIANTS}

    rows = []
    for name in VARIANTS:
        run = runs_by_name.get(name)
        if run is None:
            continue
        history = list(run.history(samples=200, keys=METRICS, pandas=False))
        last = {m: None for m in METRICS}
        traj = []
        for row in history:
            if row.get("eval/loss") is not None:
                traj.append((row.get("_step"), row.get("eval/loss"), row.get("eval/ppl"), row.get("eval/mask_acc")))
            for m in METRICS:
                if row.get(m) is not None:
                    last[m] = row[m]
        rows.append(
            {
                "name": name,
                "label": VARIANTS[name]["label"],
                "params": VARIANTS[name]["params"],
                "last": last,
                "runtime_s": run.summary.get("_runtime"),
                "trajectory": traj,
                "state": run.state,
            }
        )

    headers = [
        "variant",
        "params",
        "state",
        "final train loss",
        "final eval loss",
        "final eval ppl",
        "final eval mask_acc",
        "runtime (min)",
    ]
    md_lines = [
        "## V2 final-step summary",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        last = row["last"]
        runtime = (row["runtime_s"] or 0) / 60.0
        md_lines.append(
            "| {label} | {params:,} | {state} | {tl} | {el} | {ep} | {ea} | {rt:.1f} |".format(
                label=row["label"],
                params=row["params"],
                state=row["state"],
                tl=_fmt(last.get("train/loss")),
                el=_fmt(last.get("eval/loss")),
                ep=_fmt(last.get("eval/ppl")),
                ea=_fmt(last.get("eval/mask_acc")),
                rt=runtime,
            )
        )

    md_lines.append("")
    md_lines.append("### Eval trajectories (every 5k steps)")
    md_lines.append("")
    md_lines.append("```")
    md_lines.append(f"{'step':>6}  " + "  ".join(f"{r['name']:>27}" for r in rows))
    # Stitch trajectories into a step-aligned grid
    all_steps = sorted({s for r in rows for (s, *_rest) in r["trajectory"]})
    by_run = {r["name"]: dict((s, (el, ep, ea)) for (s, el, ep, ea) in r["trajectory"]) for r in rows}
    for step in all_steps:
        cells = []
        for r in rows:
            v = by_run[r["name"]].get(step)
            cells.append(f"{(v[0] if v else float('nan')):>27.4f}")
        md_lines.append(f"{step:>6}  " + "  ".join(cells))
    md_lines.append("```")

    text = "\n".join(md_lines)
    print(text)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(text + "\n")
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
