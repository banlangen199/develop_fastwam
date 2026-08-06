from __future__ import annotations

import torch
from omegaconf import OmegaConf

from action_dream_alpha.calibrate import (
    alpha_metrics,
    build_episode_calibration_split,
    choose_alpha,
    compute_dense_record,
    run_summary,
)


def test_dense_reduction_uses_full_mixed_distribution_and_expected_order() -> None:
    probs = torch.tensor(
        [[[[0.10, 0.10, 0.05, 0.15, 0.30, 0.30], [0.20, 0.10, 0.25, 0.05, 0.20, 0.20]],
          [[0.10, 0.20, 0.10, 0.10, 0.25, 0.25], [0.10, 0.20, 0.20, 0.10, 0.20, 0.20]]]]
    )
    result = compute_dense_record(
        probs,
        {"video": [0, 2], "dream": [2, 4], "action": [4, 6]},
        n_valid=6,
    )
    expected = (probs[..., 2:4] * 6).mean(dim=1).amax(dim=1)
    torch.testing.assert_close(result["score"], expected)
    torch.testing.assert_close(result["source_mass"].sum(dim=-1), torch.ones(1))
    assert result["probability_sum_error"].item() < 1e-6


def test_alpha_zero_keeps_everything_and_k_is_monotonic() -> None:
    score = torch.tensor([[0.2, 0.8, 1.1], [0.0, 1.5, 2.0]])
    mass = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.1, 0.1]])
    metrics = [alpha_metrics(score, mass, alpha) for alpha in (0.0, 0.5, 1.0, 2.5)]
    assert metrics[0]["mean_k"] == 3.0
    assert metrics[0]["mass_retention"] == 1.0
    assert [item["mean_k"] for item in metrics] == sorted(
        [item["mean_k"] for item in metrics], reverse=True
    )
    assert metrics[-1]["k_zero_ratio_diagnostic_only"] == 1.0


def test_selection_ignores_k_zero_ratio_and_minimizes_mean_k() -> None:
    rows = [
        {
            "alpha": 0.5,
            "mean_k": 120.0,
            "mass_retention": 0.98,
            "mean_relative_action_loss_delta": 0.005,
            "k_zero_ratio_diagnostic_only": 0.0,
        },
        {
            "alpha": 1.0,
            "mean_k": 90.0,
            "mass_retention": 0.96,
            "mean_relative_action_loss_delta": 0.009,
            "k_zero_ratio_diagnostic_only": 0.75,
        },
        {
            "alpha": 1.5,
            "mean_k": 60.0,
            "mass_retention": 0.90,
            "mean_relative_action_loss_delta": 0.002,
            "k_zero_ratio_diagnostic_only": 0.0,
        },
    ]
    selected = choose_alpha(
        rows, min_mass_retention=0.95, max_relative_action_loss_delta=0.01
    )
    assert selected["selected_alpha"] == 1.0
    assert selected["selection_rule"]["k_zero_ratio_used_for_selection"] is False


def test_episode_split_is_deterministic_balanced_and_uses_uniform_frames() -> None:
    records = []
    for task in ("a", "b"):
        for episode in range(4):
            for frame in range(10):
                records.append(
                    {
                        "task": task,
                        "dataset_index": 0,
                        "episode_index": episode + (100 if task == "b" else 0),
                        "frame_index": frame,
                        "sample_index": len(records),
                    }
                )
    first = build_episode_calibration_split(
        records,
        episode_ratio=0.25,
        max_episodes_per_task=1,
        observations_per_episode=3,
        seed=42,
    )
    second = build_episode_calibration_split(
        records,
        episode_ratio=0.25,
        max_episodes_per_task=1,
        observations_per_episode=3,
        seed=42,
    )
    assert first == second
    assert len(first) == 6
    assert {item["task"] for item in first} == {"a", "b"}
    for task in ("a", "b"):
        frames = [item["frame_index"] for item in first if item["task"] == task]
        assert frames == [0, 4, 9]


def test_summary_writes_recommendation_and_diagnostic_k_zero(tmp_path) -> None:
    shard_dir = tmp_path / "profile_shards"
    shard_dir.mkdir()
    score = torch.tensor(
        [
            [[0.5, 1.5, 2.0], [0.4, 1.4, 1.8]],
            [[0.6, 1.6, 2.1], [0.3, 1.3, 1.7]],
        ],
        dtype=torch.float16,
    )
    mass = torch.full_like(score, 0.1)
    torch.save(
        {
            "score": score,
            "dream_token_mass": mass,
            "source_mass": torch.tensor(
                [[[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]], [[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]]]
            ),
            "probability_sum_error": torch.zeros(2, 2),
            "layer_ids": torch.tensor([0, 1]),
            "diffusion_step_indices": [0, 4],
            "metadata": {"task": "synthetic"},
        },
        shard_dir / "sample_000000.pt",
    )
    (tmp_path / "paired_action_loss.csv").write_text(
        "alpha,relative_action_loss_delta\n0.0,0.0\n1.0,0.005\n",
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "alpha_values": [0.0, 1.0],
            "selection": {
                "min_mass_retention": 0.6,
                "max_relative_action_loss_delta": 0.01,
            },
        }
    )
    run_summary(cfg, tmp_path)
    recommendation = __import__("json").loads(
        (tmp_path / "recommended_alpha.json").read_text(encoding="utf-8")
    )
    assert recommendation["selected_alpha"] == 1.0
    assert recommendation["selection_rule"]["k_zero_ratio_used_for_selection"] is False
    assert (tmp_path / "alpha_sweep.csv").exists()
    assert (tmp_path / "dense_summary.json").exists()
