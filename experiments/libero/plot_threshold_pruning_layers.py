"""Plot one layer-pruning heatmap for every LIBERO policy replan."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to *_pruning_layers.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <CSV stem>_plots next to the CSV)",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def _load_replans(path: Path) -> dict[tuple[int, int], list[dict[str, str]]]:
    replans: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "episode",
            "success",
            "replan",
            "denoise_step",
            "layer",
            "prune_ratio",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        for row in reader:
            key = (int(row["episode"]), int(row["replan"]))
            replans[key].append(row)
    if not replans:
        raise ValueError(f"No pruning records found in {path}")
    return dict(replans)


def _matrix_for_replan(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, list[int], list[int]]:
    steps = sorted({int(row["denoise_step"]) for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})
    matrix = np.full((len(steps), len(layers)), np.nan, dtype=np.float64)
    step_index = {value: index for index, value in enumerate(steps)}
    layer_index = {value: index for index, value in enumerate(layers)}
    for row in rows:
        position = (step_index[int(row["denoise_step"])], layer_index[int(row["layer"])])
        if not np.isnan(matrix[position]):
            raise ValueError(
                "Duplicate pruning record for "
                f"denoise_step={row['denoise_step']}, layer={row['layer']}"
            )
        matrix[position] = float(row["prune_ratio"])
    if np.isnan(matrix).any():
        raise ValueError("A replan does not contain the full denoise-step/layer grid.")
    return matrix, steps, layers


def _plot_replan(
    *,
    rows: list[dict[str, str]],
    episode: int,
    replan: int,
    output_path: Path,
    dpi: int,
) -> None:
    matrix, steps, layers = _matrix_for_replan(rows)
    success = rows[0]["success"].strip().lower() == "true"
    alpha = float(rows[0]["alpha"])

    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    image = ax.imshow(
        matrix * 100.0,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=100.0,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Pruned Dream tokens (%)")
    ax.set_xticks(np.arange(len(layers)), labels=layers)
    ax.set_yticks(np.arange(len(steps)), labels=steps)
    ax.set_xlabel("MoT layer")
    ax.set_ylabel("Denoise step")
    ax.set_title(
        f"Episode {episode} | Replan {replan} | "
        f"success={success} | alpha={alpha:g} | mean pruned={matrix.mean():.1%}"
    )
    ax.set_xticks(np.arange(-0.5, len(layers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(steps), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.25, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    csv_path = args.csv_path.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else csv_path.with_name(f"{csv_path.stem}_plots")
    )
    replans = _load_replans(csv_path)
    for (episode, replan), rows in sorted(replans.items()):
        output_path = output_dir / f"episode_{episode:02d}" / f"replan_{replan:03d}.png"
        _plot_replan(
            rows=rows,
            episode=episode,
            replan=replan,
            output_path=output_path,
            dpi=args.dpi,
        )
    print(f"Wrote {len(replans)} replan plots to {output_dir}")


if __name__ == "__main__":
    main()
