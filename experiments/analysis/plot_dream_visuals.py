#!/usr/bin/env python3
"""Figures for the Dream branch of the finished RoutedWAM run.

Reads the payloads written by `dump_dream_visuals.py` and emits four artefacts:

``dream_predictions.png``
    What the Dream expert actually imagines: its depth and DINO predictions for
    t+16 and t+32, beside the ground truth the online extractors compute from
    the same clip's future frames.

``dream_router_heatmap.png``
    The router's decision surface -- gate and keep for all 72 Dream tokens at
    all 30 layers, plus the per-group and per-layer marginals.

``dream_action_collaboration.png``
    Where the action queries spend their attention. Keeping a token and using it
    are different things, so the kept-ratio and the received-attention are shown
    against each other.

``dream_process.gif``
    The gate is a function of the action queries, so it moves as the action
    chunk is denoised. One frame per action denoising step.

Usage:
    python experiments/analysis/plot_dream_visuals.py \
        --in evaluate_results/dream_visuals --out evaluate_results/dream_visuals
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

# Same validated slots as plot_router_curves.py, so the two figure sets read as
# one family.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3e0"
SURFACE = "#fcfcfb"

TOKENS_PER_GROUP = 9


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def short_group(name: str) -> str:
    """'depth@t0/primary' -> 'depth t0 prim'."""
    body = name.replace("@", " ").replace("/", " ")
    return body.replace("primary", "prim").replace("wrist", "wrst")


def task_text(prompt: str, width: int = 38) -> str:
    """Strip the fixed LIBERO preamble and wrap what is left."""
    import textwrap

    marker = "following instruction:"
    instruction = prompt.split(marker)[-1].strip() if marker in prompt else prompt
    return "\n".join(textwrap.wrap(instruction, width=width)[:3])


def load(in_dir: Path) -> list[dict[str, Any]]:
    files = sorted(in_dir.glob("sample_*.pt"))
    if not files:
        raise SystemExit(f"no sample_*.pt under {in_dir}")
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


def rgb_image(frame: torch.Tensor) -> np.ndarray:
    """[3,H,W] in (-1,1) -> HWC in [0,1]."""
    return ((frame.permute(1, 2, 0).numpy() + 1.0) * 0.5).clip(0.0, 1.0)


def dino_pca(reference: torch.Tensor, *others: torch.Tensor) -> list[np.ndarray]:
    """Project DINO grids onto the GT's top-3 principal directions.

    Fitting on the ground truth and reusing the basis for the prediction is what
    makes the two images comparable: a per-image PCA would hide a prediction
    that is right up to a rotation and, worse, make a wrong one look plausible.
    """
    flat = reference.reshape(-1, reference.shape[-1]).float()
    mean = flat.mean(dim=0, keepdim=True)
    _, _, vectors = torch.pca_lowrank(flat - mean, q=3)
    scores = (flat - mean) @ vectors
    low = scores.quantile(0.02, dim=0)
    high = scores.quantile(0.98, dim=0)

    def project(grid: torch.Tensor) -> np.ndarray:
        shape = grid.shape[:-1]
        value = (grid.reshape(-1, grid.shape[-1]).float() - mean) @ vectors
        value = ((value - low) / (high - low).clamp(min=1e-6)).clamp(0, 1)
        return value.reshape(*shape, 3).numpy()

    return [project(reference), *[project(other) for other in others]]


# --------------------------------------------------------------------- figure 1
def figure_predictions(samples: list[dict], out: Path, max_samples: int = 4) -> None:
    shown = samples[:max_samples]
    rows = 2 * len(shown)
    fig, axes = plt.subplots(rows, 5, figsize=(16.0, 2.45 * rows), facecolor=SURFACE)
    axes = np.atleast_2d(axes)

    for index, sample in enumerate(shown):
        offsets = sample["offsets"]
        depth_gt = sample["depth_gt"]
        depth_pred = sample["depth_pred"]
        dino_grids = dino_pca(sample["dino_gt"], sample["dino_pred"])
        dino_gt, dino_pred = dino_grids[0], dino_grids[1]

        # One scale for GT and prediction: separate colour limits would make a
        # flat prediction look like a detailed one.
        low = float(depth_gt.quantile(0.02))
        high = float(depth_gt.quantile(0.98))

        for row_offset, kind in enumerate(("depth", "dino")):
            row = 2 * index + row_offset
            ax = axes[row][0]
            ax.imshow(rgb_image(sample["rgb"]["t0"]))
            ax.set_ylabel(kind, fontsize=9, color=INK)
            if row_offset == 0:
                ax.set_title("input t0  (primary | wrist)", fontsize=7.5, color=INK2, loc="left")
                # The instruction goes under the frame: as a title it runs past
                # the panel and overprints the next column's title.
                ax.set_xlabel(task_text(sample["prompt"]), fontsize=7, color=INK, labelpad=4)
            else:
                ax.set_title("input t0", fontsize=7.5, color=INK2, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])

            for horizon in range(len(offsets)):
                valid = bool(sample["future_valid"][horizon])
                if kind == "depth":
                    gt_img, pred_img = depth_gt[horizon].numpy(), depth_pred[horizon].numpy()
                    kwargs = {"cmap": "magma", "vmin": low, "vmax": high}
                    error = float((depth_pred[horizon] - depth_gt[horizon]).abs().mean())
                    score = f"L1={error:.3f}"
                else:
                    gt_img, pred_img = dino_gt[horizon], dino_pred[horizon]
                    kwargs = {}
                    cos = torch.nn.functional.cosine_similarity(
                        sample["dino_pred"][horizon].reshape(-1, 768),
                        sample["dino_gt"][horizon].reshape(-1, 768),
                        dim=-1,
                    ).mean()
                    score = f"cos={float(cos):.3f}"

                ax_gt = axes[row][1 + 2 * horizon]
                ax_pred = axes[row][2 + 2 * horizon]
                ax_gt.imshow(gt_img, **kwargs)
                ax_pred.imshow(pred_img, **kwargs)
                tag = "" if valid else "  [clip ended: GT is a repeat]"
                ax_gt.set_title(
                    f"ground truth  t+{offsets[horizon]}{tag}",
                    fontsize=7.5, color=INK2, loc="left",
                )
                ax_pred.set_title(
                    f"dream prediction  t+{offsets[horizon]}   {score}",
                    fontsize=7.5, color=PALETTE[0], loc="left",
                )
                for a in (ax_gt, ax_pred):
                    a.set_xticks([])
                    a.set_yticks([])

    fig.suptitle(
        "RoutedWAM dream expert — imagined futures vs. the clip's real futures "
        "(step 12500, 4-suite LIBERO)",
        fontsize=13, color=INK, x=0.012, ha="left", y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"-> {out}")
    plt.close(fig)


# --------------------------------------------------------------------- figure 2
def group_separators(ax: plt.Axes, names: list[str], total: int) -> None:
    for boundary in range(TOKENS_PER_GROUP, total, TOKENS_PER_GROUP):
        ax.axvline(boundary - 0.5, color="#ffffff", linewidth=1.1)
    centres = [i * TOKENS_PER_GROUP + TOKENS_PER_GROUP / 2 - 0.5 for i in range(len(names))]
    ax.set_xticks(centres)
    ax.set_xticklabels([short_group(n) for n in names], fontsize=6.4, color=INK2, rotation=30, ha="right")


def figure_router(samples: list[dict], out: Path) -> None:
    names = samples[0]["group_names"]
    group_ids = samples[0]["group_ids"].numpy()
    num_tokens = len(group_ids)

    # [samples, steps, layers, tokens]
    gate = torch.stack([s["gate"] for s in samples]).numpy()
    keep = torch.stack([s["keep"].float() for s in samples]).numpy()

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.4), facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.5, wspace=0.22, top=0.88, bottom=0.1)

    ax = axes[0][0]
    image = gate.mean(axis=(0, 1))
    handle = ax.imshow(image, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
    group_separators(ax, names, num_tokens)
    ax.set_ylabel("MoT layer", fontsize=8, color=INK2)
    ax.set_title(
        "Soft gate per dream token  (mean over samples and action steps)",
        fontsize=9.5, color=INK, loc="left", pad=10,
    )
    fig.colorbar(handle, ax=ax, fraction=0.03, pad=0.015)

    ax = axes[0][1]
    image = keep.mean(axis=(0, 1))
    handle = ax.imshow(image, aspect="auto", cmap="cividis", vmin=0.0, vmax=1.0, origin="lower")
    group_separators(ax, names, num_tokens)
    ax.set_ylabel("MoT layer", fontsize=8, color=INK2)
    ax.set_title(
        "Hard keep decision  (1 = action expert may read this token)",
        fontsize=9.5, color=INK, loc="left", pad=10,
    )
    fig.colorbar(handle, ax=ax, fraction=0.03, pad=0.015)

    ax = axes[1][0]
    style(ax)
    per_group_keep = [keep[..., group_ids == i].mean() for i in range(len(names))]
    per_group_gate = [gate[..., group_ids == i].mean() for i in range(len(names))]
    positions = np.arange(len(names))
    ax.bar(positions - 0.19, per_group_keep, width=0.36, color=PALETTE[0], label="keep ratio")
    ax.bar(positions + 0.19, per_group_gate, width=0.36, color=PALETTE[1], label="mean gate")
    ax.axhline(0.25, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate(
        "target keep* = 0.25", xy=(-0.4, 0.258), fontsize=7, color=INK2, ha="left"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([short_group(n) for n in names], fontsize=6.8, color=INK2, rotation=30, ha="right")
    ax.set_ylabel("ratio", fontsize=8, color=INK2)
    ax.set_title(
        "Per-group selection  (differentiated groups = routing, flat = sparsifying)",
        fontsize=9.5, color=INK, loc="left", pad=10,
    )
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, ncol=2, loc="upper left")
    ax.grid(axis="x", visible=False)

    ax = axes[1][1]
    style(ax)
    for index, sample in enumerate(samples[:5]):
        per_layer = sample["keep"].float().mean(dim=(0, 2)).numpy()
        ax.plot(
            per_layer, color=PALETTE[index % len(PALETTE)], linewidth=1.8,
            label=f"sample {index}", solid_capstyle="round",
        )
    ax.plot(keep.mean(axis=(0, 1, 3)), color=INK, linewidth=2.6, label="mean", zorder=5)
    ax.set_xlabel("MoT layer", fontsize=8, color=INK2)
    ax.set_ylabel("keep ratio", fontsize=8, color=INK2)
    ax.set_title("Depth profile: how much imagination survives per layer",
                 fontsize=9.5, color=INK, loc="left", pad=10)
    ax.set_ylim(-0.03, 1.28)
    ax.legend(frameon=False, fontsize=7.2, labelcolor=INK2, ncol=4, loc="upper left")

    fig.suptitle(
        f"RoutedWAM imagination router — decision surface over {num_tokens} dream tokens "
        f"× 30 layers  (overall keep={keep.mean():.4f}, gate={gate.mean():.4f})",
        fontsize=13, color=INK, x=0.012, ha="left", y=0.965,
    )
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"-> {out}")
    plt.close(fig)


# --------------------------------------------------------------------- figure 3
def figure_collaboration(samples: list[dict], out: Path) -> None:
    names = samples[0]["group_names"]
    group_ids = samples[0]["group_ids"].numpy()

    probs = torch.stack([s["dream_probs"] for s in samples]).numpy()  # [S,steps,L,Sd]
    keep = torch.stack([s["keep"].float() for s in samples]).numpy()
    gate = torch.stack([s["gate"] for s in samples]).numpy()
    mass = {
        key: torch.stack([s[f"mass_{key}"] for s in samples]).numpy()
        for key in ("video", "dream", "action")
    }

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.2), facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.5, wspace=0.24, top=0.88, bottom=0.1)

    ax = axes[0][0]
    style(ax)
    layers = np.arange(mass["dream"].shape[-1])
    bottom = np.zeros_like(layers, dtype=float)
    for index, key in enumerate(("video", "dream", "action")):
        values = mass[key].mean(axis=(0, 1))
        ax.bar(layers, values, bottom=bottom, width=0.92,
               color=PALETTE[index], label=f"{key} keys", linewidth=0)
        bottom += values
    ax.set_xlabel("MoT layer", fontsize=8, color=INK2)
    ax.set_ylabel("share of action attention", fontsize=8, color=INK2)
    ax.set_ylim(0, 1.22)
    ax.set_title(
        "What the action expert reads  (attention mass by key source)",
        fontsize=9.5, color=INK, loc="left", pad=10,
    )
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)

    ax = axes[0][1]
    style(ax)
    received = [probs[..., group_ids == i].sum(axis=-1).mean() for i in range(len(names))]
    kept = [keep[..., group_ids == i].mean() for i in range(len(names))]
    positions = np.arange(len(names))
    ax.bar(positions - 0.19, kept, width=0.36, color=PALETTE[0], label="keep ratio (allowed)")
    twin = ax.twinx()
    twin.bar(positions + 0.19, received, width=0.36, color=PALETTE[2], label="attention received")
    twin.set_ylabel("attention mass", fontsize=8, color=PALETTE[2])
    twin.tick_params(colors=PALETTE[2], labelsize=8)
    twin.grid(False)
    ax.set_xticks(positions)
    ax.set_xticklabels([short_group(n) for n in names], fontsize=6.8, color=INK2, rotation=30, ha="right")
    ax.set_ylabel("keep ratio", fontsize=8, color=PALETTE[0])
    ax.set_title(
        "Allowed vs. actually used, per group  (a kept token can still be ignored)",
        fontsize=9.5, color=INK, loc="left", pad=10,
    )
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    ax.grid(axis="x", visible=False)

    ax = axes[1][0]
    style(ax)
    flat_gate = gate.mean(axis=(0, 1)).reshape(-1)
    flat_probs = probs.mean(axis=(0, 1)).reshape(-1)
    token_groups = np.tile(group_ids, gate.shape[2])
    for index in range(len(names)):
        selector = token_groups == index
        ax.scatter(
            flat_gate[selector], flat_probs[selector], s=9, alpha=0.75,
            color=plt.get_cmap("tab10")(index % 10), label=short_group(names[index]),
            linewidths=0,
        )
    ax.set_yscale("log")
    ax.set_xlabel("gate (router's soft decision)", fontsize=8, color=INK2)
    ax.set_ylabel("attention received", fontsize=8, color=INK2)
    ax.set_title(
        "Does a high gate mean a used token?  (one point = one token at one layer)",
        fontsize=9.5, color=INK, loc="left", pad=10,
    )
    ax.legend(frameon=False, fontsize=6.2, labelcolor=INK2, ncol=4, loc="lower right")

    ax = axes[1][1]
    per_group_layer = np.stack(
        [probs[..., group_ids == i].sum(axis=-1).mean(axis=(0, 1)) for i in range(len(names))],
        axis=1,
    )
    handle = ax.imshow(per_group_layer, aspect="auto", cmap="rocket" if "rocket" in plt.colormaps() else "inferno", origin="lower")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([short_group(n) for n in names], fontsize=6.4, color=INK2, rotation=30, ha="right")
    ax.set_ylabel("MoT layer", fontsize=8, color=INK2)
    ax.set_title(
        "Dream contribution to the action, per group and depth",
        fontsize=9.5, color=INK, loc="left", pad=10,
    )
    fig.colorbar(handle, ax=ax, fraction=0.035, pad=0.015)

    total_dream = mass["dream"].mean()
    fig.suptitle(
        "RoutedWAM dream ↔ action collaboration — "
        f"{100 * total_dream:.2f}% of action attention lands on imagination tokens",
        fontsize=13, color=INK, x=0.012, ha="left", y=0.965,
    )
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"-> {out}")
    plt.close(fig)


# --------------------------------------------------------------------- figure 4
def animation_process(samples: list[dict], out: Path) -> None:
    sample = samples[0]
    names = sample["group_names"]
    group_ids = sample["group_ids"].numpy()
    gate = sample["gate"].numpy()          # [steps, layers, tokens]
    keep = sample["keep"].float().numpy()
    probs = sample["dream_probs"].numpy()
    steps = gate.shape[0]

    fig = plt.figure(figsize=(14.6, 4.3), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.3,
                            left=0.05, right=0.985, top=0.80, bottom=0.22)
    ax_gate = fig.add_subplot(grid[0])
    ax_group = fig.add_subplot(grid[1])
    ax_mass = fig.add_subplot(grid[2])

    image = ax_gate.imshow(
        gate[0], aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, origin="lower"
    )
    group_separators(ax_gate, names, gate.shape[-1])
    ax_gate.set_ylabel("MoT layer", fontsize=8, color=INK2)
    ax_gate.set_title("gate per dream token", fontsize=9.5, color=INK, loc="left", pad=8)
    fig.colorbar(image, ax=ax_gate, fraction=0.03, pad=0.015)

    style(ax_group)
    positions = np.arange(len(names))
    bars = ax_group.bar(positions, np.zeros(len(names)), width=0.68, color=PALETTE[0])
    ax_group.axhline(0.25, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax_group.set_xticks(positions)
    ax_group.set_xticklabels(
        [short_group(n) for n in names], fontsize=6.4, color=INK2, rotation=30, ha="right"
    )
    ax_group.set_ylim(0, 1.02)
    ax_group.set_ylabel("keep ratio", fontsize=8, color=INK2)
    ax_group.set_title("per-group keep", fontsize=9.5, color=INK, loc="left", pad=8)
    ax_group.grid(axis="x", visible=False)

    style(ax_mass)
    layers = np.arange(gate.shape[1])
    (line_mass,) = ax_mass.plot([], [], color=PALETTE[2], linewidth=2.0)
    ax_mass.set_xlim(0, gate.shape[1] - 1)
    ax_mass.set_ylim(0, float(probs.sum(axis=-1).max()) * 1.15 + 1e-6)
    ax_mass.set_xlabel("MoT layer", fontsize=8, color=INK2)
    ax_mass.set_ylabel("dream attention mass", fontsize=8, color=INK2)
    ax_mass.set_title("imagination actually read", fontsize=9.5, color=INK, loc="left", pad=8)

    title = fig.suptitle("", fontsize=12.5, color=INK, x=0.012, ha="left", y=0.955)

    def update(step: int):
        image.set_data(gate[step])
        values = [keep[step][:, group_ids == i].mean() for i in range(len(names))]
        for bar, value in zip(bars, values):
            bar.set_height(value)
        line_mass.set_data(layers, probs[step].sum(axis=-1))
        title.set_text(
            f"Action denoising step {step + 1}/{steps} — the router re-decides every step  "
            f"(keep={keep[step].mean():.3f})"
        )
        return [image, line_mass, title, *bars]

    animation = FuncAnimation(fig, update, frames=steps, blit=False)
    animation.save(out, writer=PillowWriter(fps=2))
    print(f"-> {out}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=Path, default=Path("evaluate_results/dream_visuals"))
    ap.add_argument("--out", type=Path, default=Path("evaluate_results/dream_visuals"))
    args = ap.parse_args()

    samples = load(args.in_dir)
    print(f"loaded {len(samples)} samples from {args.in_dir}")
    args.out.mkdir(parents=True, exist_ok=True)

    figure_predictions(samples, args.out / "dream_predictions.png")
    figure_router(samples, args.out / "dream_router_heatmap.png")
    figure_collaboration(samples, args.out / "dream_action_collaboration.png")
    animation_process(samples, args.out / "dream_process.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
