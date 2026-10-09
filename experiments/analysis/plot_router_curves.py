#!/usr/bin/env python3
"""Training curves for the RoutedWAM router ablation.

Answers two questions that a scrolled log cannot:

1. *Does the router actually select anything?*  `router_keep_ratio` is the hard
   decision (fraction of imagination tokens the action expert still attends to);
   `router_gate_mean` is the soft gate it is thresholded from.  They are not
   interchangeable -- during warmup the gate falls while keep_ratio is pinned at
   exactly 1.0, which reads as "nothing is happening" if you only look at one.

2. *Is `loss_router_budget=0.0000` a dead loss or a converged one?*  The term is
   ``lambda * (mean_gate - target_keep_ratio)^2``.  With lambda=0.01 and the gate
   converged onto the target, its true value is ~1e-5 -- which prints as 0.0000 at
   four decimals.  Panel 3 is log-scaled precisely so that floor is visible as a
   descent rather than as a disappearance.

Usage:
    python experiments/analysis/plot_router_curves.py \
        --watch-dir evaluate_results/watch \
        --run run_a="learned, keep*=0.25" --run run_b="threshold" \
        --out evaluate_results/router_curves.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_train_log import parse_file  # noqa: E402

# Validated categorical slots 1-5 (see dataviz references/palette.md); assigned in
# fixed order, never cycled. Three of them sit under 3:1 on the light surface, so
# the contrast WARN is relieved by direct end-labels plus distinct dashes.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DASHES = [(None, None), (6, 2), (2, 2), (6, 2, 2, 2), (1, 2)]

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3e0"
SURFACE = "#fcfcfb"

# Run ids are site-specific, so there is no default table: pass one `--run
# <dir>[=<label>]` per curve. Each <dir> is a directory under `--watch-dir`
# holding a `job_log.txt` (the training stdout, however your launcher captured
# it); `parse_train_log.py` scrapes the metrics out of it.
RUNS: List[tuple[str, str]] = []

PANELS = [
    ("router_keep_ratio", "Hard selection: kept imagination tokens", "keep ratio", False),
    ("router_gate_mean", "Soft gate (pre-threshold)", "mean gate", False),
    ("loss_router_budget", "Budget loss  λ·(ḡ−keep*)²", "loss", True),
    ("loss_action", "Action flow-matching loss", "loss", True),
    ("loss_dream", "Dream regression loss", "loss", True),
]

# The groups the router scores separately. `modality_horizon_camera` granularity,
# so eight slots: {depth,dino} x {t0,t1} x {primary,wrist}.
GROUPS = [
    f"router_keep_{m}@t{t}/{c}"
    for m in ("depth", "dino")
    for t in (0, 1)
    for c in ("primary", "wrist")
]


def group_label(key: str) -> str:
    body = key.replace("router_keep_", "")
    modality, rest = body.split("@")
    horizon, camera = rest.split("/")
    return f"{modality}\n{horizon}·{camera[:4]}"


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-dir", type=Path, default=Path("evaluate_results/watch"))
    ap.add_argument("--out", type=Path, default=Path("evaluate_results/router_curves.png"))
    ap.add_argument(
        "--run",
        "--job",
        action="append",
        default=None,
        dest="job",
        metavar="RUN_DIR[=LABEL]",
        help="a run directory under --watch-dir to plot; repeatable",
    )
    ap.add_argument("--title", default=None, help="override the figure title")
    args = ap.parse_args()

    specs = args.job or RUNS
    if not specs:
        ap.error("at least one --run RUN_DIR[=LABEL] is required")
    runs = []
    for spec in specs:
        job, label = spec if isinstance(spec, tuple) else spec.partition("=")[::2]
        runs.append((job, label or job))

    series: Dict[str, List[dict]] = {}
    for job, label in runs:
        log = args.watch_dir / job / "job_log.txt"
        if not log.exists():
            print(f"skip (no log): {label}")
            continue
        rows = parse_file(log)
        if not rows:
            # A job that has not logged a step yet is the normal case for the
            # first few minutes; saying so beats an empty figure.
            print(f"skip (no steps logged yet): {label}")
            continue
        series[label] = rows

    if not series:
        print("no logs found", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.6), facecolor=SURFACE)
    fig.subplots_adjust(right=0.985, left=0.05, hspace=0.46, wspace=0.28,
                        top=0.86, bottom=0.17)

    warm = None  # router warmup end: 10% of total steps
    for panel_idx, (key, title, ylab, logy) in enumerate(PANELS):
        ax = axes.flat[panel_idx]
        style(ax)
        ends: List[tuple[float, str, str]] = []
        for idx, (label, rows) in enumerate(series.items()):
            xs = [r["step"] for r in rows if isinstance(r.get(key), float)]
            ys = [r[key] for r in rows if isinstance(r.get(key), float)]
            if not xs:
                continue
            if warm is None and rows and "total" in rows[0]:
                warm = 0.1 * float(rows[0]["total"])
            if logy:
                # A converged budget term reaches ~1e-5; clamping keeps it on a
                # log axis instead of dropping the tail silently.
                ys = [max(y, 1e-6) for y in ys]
            dash = DASHES[idx % len(DASHES)]
            colour = PALETTE[idx % len(PALETTE)]
            (line,) = ax.plot(
                xs, ys, color=colour, linewidth=2,
                solid_capstyle="round", label=label,
            )
            if dash[0] is not None:
                line.set_dashes(list(dash))
            ends.append((ys[-1], label.split()[0], colour))

        # Direct end-labels are the relief the palette's contrast WARN requires,
        # so they must stay legible: nudge apart any that would overprint.
        ends.sort()
        lo, hi = ax.get_ylim()
        if logy:
            import math

            span = math.log10(max(hi, 1e-9)) - math.log10(max(lo, 1e-9))
            positions = [math.log10(max(v, 1e-9)) for v, _, _ in ends]
        else:
            span = hi - lo
            positions = [v for v, _, _ in ends]
        min_gap = 0.055 * span
        for i in range(1, len(positions)):
            if positions[i] - positions[i - 1] < min_gap:
                positions[i] = positions[i - 1] + min_gap
        for (value, tag, colour), pos in zip(ends, positions):
            ax.annotate(
                tag,
                xy=(1.005, (10**pos if logy else pos)),
                xycoords=("axes fraction", "data"),
                color=colour, fontsize=7.5, va="center", fontweight="bold",
                annotation_clip=False,
            )
        if warm:
            ax.axvspan(0, warm, color="#000000", alpha=0.05, linewidth=0)
            ax.annotate(
                "warmup", xy=(warm, 1.015), xycoords=("data", "axes fraction"),
                ha="center", fontsize=6.8, color=INK2,
            )
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=10)
        ax.set_xlabel("step", fontsize=8, color=INK2)
        ax.set_ylabel(ylab, fontsize=8, color=INK2)

    # ----------------------------------------------------------------- panel 6
    # The whole point of a *router* is that it treats groups differently. A flat
    # profile here means the learned gate has collapsed to a global sparsity
    # knob, which is a materially weaker claim than "it routes".
    ax = axes.flat[5]
    style(ax)
    width = 0.8 / max(len(series), 1)
    for idx, (label, rows) in enumerate(series.items()):
        final = rows[-1]
        vals = [final.get(g) for g in GROUPS]
        xs = [i + idx * width - 0.4 + width / 2 for i, v in enumerate(vals) if isinstance(v, float)]
        ys = [v for v in vals if isinstance(v, float)]
        if not ys:
            continue
        ax.bar(
            xs, ys, width=width * 0.86, color=PALETTE[idx % len(PALETTE)],
            label=label, linewidth=0.8, edgecolor=SURFACE,
        )
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([group_label(g) for g in GROUPS], fontsize=6.3, color=INK2)
    ax.set_title(
        "Per-group selection at step 416  (is it routing, or just sparsifying?)",
        fontsize=9.5, color=INK, loc="left", pad=10,
    )
    ax.set_ylabel("keep ratio", fontsize=8, color=INK2)
    ax.grid(axis="x", visible=False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.012),
        frameon=False, fontsize=8.5, labelcolor=INK2, ncol=max(len(labels), 1),
        columnspacing=2.2, handlelength=2.6,
    )
    fig.suptitle(
        args.title or "RoutedWAM imagination router — libero_goal, 416 steps, 2×8 GPU",
        fontsize=13, color=INK, x=0.012, ha="left", y=0.975,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
