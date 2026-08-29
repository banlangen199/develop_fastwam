"""Summarize 16D proxy attention against the full Action-Dream teacher.

The input is one file, or a directory containing files, written by
``eval_libero_threshold_single.py`` with proxy analysis and detailed tensor
capture enabled. Correlations are computed offline so model forward has no
SciPy/Pandas dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


METRICS = (
    "recall",
    "precision",
    "jaccard",
    "agreement",
    "full_k",
    "proxy_k",
    "k_delta",
    "false_negative_count",
    "false_positive_count",
    "score_mae",
    "score_rmse",
    "score_pearson",
    "score_spearman",
    "full_dense_dream_mass",
    "full_teacher_kept_mass",
    "proxy_mask_kept_full_mass",
    "proxy_full_mass_retention",
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0
    return float(numerator / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks (zero-based), including ties."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = 0.5 * (start + end - 1)
        start = end
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("Correlation inputs must be non-empty and have identical shapes.")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if np.allclose(left, right, rtol=0.0, atol=0.0) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _safe_pearson(_average_ranks(left), _average_ranks(right))


def _tensor_1d(record: dict[str, Any], key: str, *, batch_size: int) -> torch.Tensor:
    value = torch.as_tensor(record[key]).detach().cpu().float().reshape(-1)
    if value.numel() != batch_size:
        raise ValueError(
            f"{key} must contain one value per sample ({batch_size}), got {value.numel()}."
        )
    return value


def _sample_rows(payload: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(payload.get("records", [])):
        full_score = torch.as_tensor(record["full_score"]).detach().cpu().float()
        proxy_score = torch.as_tensor(record["proxy_score"]).detach().cpu().float()
        full_keep = torch.as_tensor(record["full_keep_mask"]).detach().cpu().bool()
        proxy_keep = torch.as_tensor(record["proxy_keep_mask"]).detach().cpu().bool()
        if full_score.ndim != 2:
            raise ValueError(f"full_score must be [B,Ndream], got {tuple(full_score.shape)}.")
        if proxy_score.shape != full_score.shape:
            raise ValueError("proxy_score shape differs from full_score.")
        if full_keep.shape != full_score.shape or proxy_keep.shape != full_score.shape:
            raise ValueError("Full/proxy masks must match the score shape.")
        batch_size = int(full_score.shape[0])
        dense_mass = _tensor_1d(
            record, "full_dense_dream_mass_per_sample", batch_size=batch_size
        )
        teacher_mass = _tensor_1d(
            record, "full_teacher_kept_mass_per_sample", batch_size=batch_size
        )
        proxy_full_mass = _tensor_1d(
            record, "proxy_mask_kept_full_mass_per_sample", batch_size=batch_size
        )

        for sample_index in range(batch_size):
            teacher = full_keep[sample_index]
            proxy = proxy_keep[sample_index]
            intersection = int((teacher & proxy).sum().item())
            union = int((teacher | proxy).sum().item())
            full_k = int(teacher.sum().item())
            proxy_k = int(proxy.sum().item())
            full_values = full_score[sample_index].numpy()
            proxy_values = proxy_score[sample_index].numpy()
            score_error = proxy_values - full_values
            row = {
                "source_file": str(source),
                "record_index": record_index,
                "task_suite": payload.get("task_suite"),
                "task_id": payload.get("task_id"),
                "episode": record.get("episode"),
                "success": record.get("success"),
                "replan": record.get("replan"),
                "denoising_step": record.get("denoise_step"),
                "layer": record.get("layer"),
                "sample_in_batch": sample_index,
                "alpha": record.get("alpha"),
                "proxy_head_dim": record.get("proxy_head_dim", payload.get("proxy_head_dim")),
                "dream_tokens": int(full_score.shape[1]),
                "full_k": full_k,
                "proxy_k": proxy_k,
                "k_delta": proxy_k - full_k,
                "recall": _safe_ratio(intersection, full_k),
                "precision": _safe_ratio(intersection, proxy_k),
                "jaccard": _safe_ratio(intersection, union),
                "agreement": float((teacher == proxy).float().mean().item()),
                "false_negative_count": int((teacher & ~proxy).sum().item()),
                "false_positive_count": int((proxy & ~teacher).sum().item()),
                "score_mae": float(np.abs(score_error).mean()),
                "score_rmse": float(np.sqrt(np.square(score_error).mean())),
                "score_pearson": _safe_pearson(full_values, proxy_values),
                "score_spearman": _safe_spearman(full_values, proxy_values),
                "full_dense_dream_mass": float(dense_mass[sample_index].item()),
                "full_teacher_kept_mass": float(teacher_mass[sample_index].item()),
                "proxy_mask_kept_full_mass": float(proxy_full_mass[sample_index].item()),
                "proxy_full_mass_retention": _safe_ratio(
                    float(proxy_full_mass[sample_index].item()),
                    float(dense_mass[sample_index].item()),
                ),
            }
            if not all(math.isfinite(float(row[key])) for key in METRICS):
                raise FloatingPointError(f"Non-finite proxy metric in {source}: {row}")
            rows.append(row)
    return rows


def _aggregate_rows(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)
    output = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        result: dict[str, Any] = {group_key: group, "count": len(group_rows)}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=np.float64)
            result[metric] = float(values.mean())
            result[f"{metric}_std"] = float(values.std())
            result[f"{metric}_min"] = float(values.min())
        output.append(result)
    return output


def _aggregate_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("task_suite"), row.get("task_id"))].append(row)
    output = []
    for (suite, task_id), task_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        result: dict[str, Any] = {
            "task_suite": suite,
            "task_id": task_id,
            "count": len(task_rows),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in task_rows], dtype=np.float64)
            result[metric] = float(values.mean())
            result[f"{metric}_std"] = float(values.std())
            result[f"{metric}_min"] = float(values.min())
        output.append(result)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary = {}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def _discover_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    paths = sorted(input_path.rglob("*_proxy_tensors.pt"))
    if not paths:
        raise FileNotFoundError(f"No *_proxy_tensors.pt files found under {input_path}.")
    return paths


def analyze_proxy_tensors(input_path: Path, output_dir: Path) -> dict[str, Any]:
    paths = _discover_inputs(input_path)
    rows: list[dict[str, Any]] = []
    record_count = 0
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            raise ValueError(f"Invalid proxy tensor payload: {path}")
        record_count += len(payload["records"])
        rows.extend(_sample_rows(payload, path))
    if not rows:
        raise RuntimeError("Proxy tensor payloads contain no records.")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "proxy_per_sample.csv", rows)
    _write_csv(output_dir / "proxy_per_layer.csv", _aggregate_rows(rows, "layer"))
    _write_csv(
        output_dir / "proxy_per_denoising_step.csv",
        _aggregate_rows(rows, "denoising_step"),
    )
    _write_csv(output_dir / "proxy_per_task.csv", _aggregate_tasks(rows))

    metrics = _metric_summary(rows)
    full_k_sum = sum(float(row["full_k"]) for row in rows)
    proxy_k_sum = sum(float(row["proxy_k"]) for row in rows)
    summary = {
        "input": str(input_path),
        "tensor_files": [str(path) for path in paths],
        "num_tensor_files": len(paths),
        "num_layer_step_records": record_count,
        "num_sample_layer_steps": len(rows),
        "proxy_head_dims": sorted({int(row["proxy_head_dim"]) for row in rows}),
        "metrics": metrics,
        "highlights": {
            "mean_recall": metrics["recall"]["mean"],
            "minimum_recall": metrics["recall"]["min"],
            "mean_jaccard": metrics["jaccard"]["mean"],
            "mean_proxy_k": metrics["proxy_k"]["mean"],
            "mean_full_k": metrics["full_k"]["mean"],
            "mean_proxy_k_over_full_k": _safe_ratio(proxy_k_sum, full_k_sum),
            "mean_full_mass_retention": metrics["proxy_full_mass_retention"]["mean"],
            "mean_score_pearson": metrics["score_pearson"]["mean"],
            "mean_score_spearman": metrics["score_spearman"]["mean"],
        },
    }
    (output_dir / "proxy_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="A proxy tensor .pt file or evaluation directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <input>/proxy_analysis for directory input.",
    )
    args = parser.parse_args()
    input_path = args.input.resolve()
    if args.output_dir is None:
        base = input_path if input_path.is_dir() else input_path.parent
        output_dir = base / "proxy_analysis"
    else:
        output_dir = args.output_dir.resolve()
    summary = analyze_proxy_tensors(input_path, output_dir)
    highlights = summary["highlights"]
    print(f"[proxy-analysis] output={output_dir}")
    print(
        "[proxy-analysis] "
        f"mean_recall={highlights['mean_recall']:.6f} "
        f"min_recall={highlights['minimum_recall']:.6f} "
        f"mean_jaccard={highlights['mean_jaccard']:.6f} "
        f"proxy/full_K={highlights['mean_proxy_k_over_full_k']:.6f} "
        f"full_mass_retention={highlights['mean_full_mass_retention']:.6f}"
    )


if __name__ == "__main__":
    main()
