from __future__ import annotations

import csv
import json

import pytest
import torch

from experiments.libero.analyze_proxy_attention import analyze_proxy_tensors


def _record(layer: int, denoise_step: int) -> dict:
    full_score = torch.tensor([[0.9, 0.8, 0.1], [0.2, 0.2, 0.2]])
    proxy_score = torch.tensor([[0.8, 0.1, 0.7], [0.2, 0.2, 0.2]])
    full_keep = torch.tensor([[True, True, False], [False, False, False]])
    proxy_keep = torch.tensor([[True, False, True], [False, False, False]])
    return {
        "episode": 0,
        "success": True,
        "replan": 0,
        "denoise_step": denoise_step,
        "layer": layer,
        "alpha": 0.5,
        "proxy_head_dim": 16,
        "proxy_channel_indices": torch.tensor(
            [0, 1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113]
        ),
        "full_score": full_score,
        "proxy_score": proxy_score,
        "full_keep_mask": full_keep,
        "proxy_keep_mask": proxy_keep,
        "full_dense_dream_mass_per_sample": torch.tensor([0.5, 0.0]),
        "full_teacher_kept_mass_per_sample": torch.tensor([0.4, 0.0]),
        "proxy_mask_kept_full_mass_per_sample": torch.tensor([0.45, 0.0]),
        "proxy_full_mass_retention_per_sample": torch.tensor([0.9, 1.0]),
    }


def test_proxy_analyzer_writes_layer_step_task_and_aggregate_outputs(tmp_path):
    input_dir = tmp_path / "eval"
    input_dir.mkdir()
    torch.save(
        {
            "task_suite": "libero_goal",
            "task_id": 3,
            "gpu_id": 0,
            "proxy_head_dim": 16,
            "records": [_record(0, 0), _record(1, 1)],
        },
        input_dir / "gpu0_task3_proxy_tensors.pt",
    )
    output_dir = tmp_path / "summary"
    summary = analyze_proxy_tensors(input_dir, output_dir)

    assert summary["num_layer_step_records"] == 2
    assert summary["num_sample_layer_steps"] == 4
    assert summary["proxy_head_dims"] == [16]
    assert summary["highlights"]["mean_recall"] == 0.75
    assert summary["highlights"]["minimum_recall"] == 0.5
    assert summary["highlights"]["mean_jaccard"] == 2.0 / 3.0
    assert summary["highlights"]["mean_full_mass_retention"] == pytest.approx(0.95)

    for name in (
        "proxy_per_sample.csv",
        "proxy_per_layer.csv",
        "proxy_per_denoising_step.csv",
        "proxy_per_task.csv",
        "proxy_summary.json",
    ):
        assert (output_dir / name).exists()

    with (output_dir / "proxy_per_layer.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["layer"]) for row in rows] == [0, 1]
    assert all("score_spearman" in row and "score_pearson" in row for row in rows)
    payload = json.loads((output_dir / "proxy_summary.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["score_spearman"]["mean"] <= 1.0
    assert payload["metrics"]["score_pearson"]["mean"] <= 1.0


def test_proxy_analyzer_correlations_and_empty_masks_never_produce_nan(tmp_path):
    path = tmp_path / "single_proxy_tensors.pt"
    record = _record(0, 0)
    torch.save({"records": [record], "proxy_head_dim": 16}, path)
    summary = analyze_proxy_tensors(path, tmp_path / "output")
    for metric in summary["metrics"].values():
        assert all(torch.isfinite(torch.tensor(value)) for value in metric.values())
    assert summary["metrics"]["recall"]["min"] == 0.5
    assert summary["metrics"]["precision"]["max"] == 1.0
    assert summary["metrics"]["jaccard"]["max"] == 1.0
