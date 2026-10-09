"""Tests for the run verifier.

The verifier's whole job is to refuse a run that looks fine but is not, so the
tests are written around the two failures actually observed in this project:

* `loss_router_budget` pinned at 0.0000 while `lambda_budget > 0`, which meant
  the gate never received pruning pressure;
* a keep ratio that never leaves `sigmoid(bias_init)`, which makes every
  per-group figure a picture of the initialisation.

A verifier that passed either of those would be worse than no verifier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.analysis.verify_run import (  # noqa: E402
    check_routing,
    check_training,
    check_visualisation,
    init_keep_ratio,
    parse_metric_series,
)


def _write_log(run_dir: Path, lines: list[str]) -> None:
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "train.log").write_text("\n".join(lines), encoding="utf-8")


def _healthy_run(tmp_path: Path, *, budget: float = 0.004, keep_end: float = 0.31) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "weights" / "step_001000.pt").write_bytes(b"x")
    (run_dir / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    lines = []
    for step in range(40):
        phase = step / 39
        lines.append(
            f"step {step} loss_action={1.0 - 0.6 * phase:.4f} loss_dream={0.8 - 0.3 * phase:.4f} "
            f"loss_router_budget={budget * (1 - 0.5 * phase):.6f} "
            f"router_keep_ratio={0.98 - (0.98 - keep_end) * phase:.4f} "
            f"router_gate_mean={0.982 - (0.982 - keep_end) * phase:.4f} "
            f"router_keep_dino@t0/primary={0.9 - 0.5 * phase:.4f} "
            f"router_keep_depth@t0/wrist={0.2 + 0.5 * phase:.4f}"
        )
    _write_log(run_dir, lines)
    return run_dir


def _config(**router):
    base = {
        "mode": "learned",
        "lambda_budget": 0.01,
        "bias_init": 4.0,
        "target_keep_ratio": 0.25,
    }
    base.update(router)
    return {
        "model": {"router": base, "loss": {"lambda_dream": 0.1}},
        "data": {"train": {"dream_target": {"enabled": True}}},
    }


def test_init_keep_ratio_matches_the_router_initialisation():
    # The router starts at sigmoid(bias_init); 4.0 -> ~0.982.
    assert init_keep_ratio(4.0) == pytest.approx(0.9820137900379085, abs=1e-9)


def test_parse_metric_series_scrapes_slashed_group_keys(tmp_path):
    run_dir = _healthy_run(tmp_path)
    series = parse_metric_series(run_dir)
    assert "loss_action" in series and len(series["loss_action"]) == 40
    # Group keys contain '/' and '@' and must survive the scrape intact.
    assert "router_keep_depth@t0/wrist" in series
    assert "router_keep_dino@t0/primary" in series


def test_healthy_run_passes_all_checks(tmp_path):
    run_dir = _healthy_run(tmp_path)
    series = parse_metric_series(run_dir)
    assert check_training(run_dir, series, _config()).passed
    assert check_routing(run_dir, series, _config()).passed


def test_dead_budget_loss_is_rejected(tmp_path):
    """The exact bug: budget identically zero while lambda_budget > 0."""
    run_dir = _healthy_run(tmp_path, budget=0.0, keep_end=0.982)
    series = parse_metric_series(run_dir)
    result = check_routing(run_dir, series, _config())
    assert not result.passed
    assert any("0.0000 for every logged step" in f for f in result.failures)


def test_keep_ratio_stuck_at_initialisation_is_rejected(tmp_path):
    run_dir = _healthy_run(tmp_path, budget=0.004, keep_end=0.982)
    series = parse_metric_series(run_dir)
    result = check_routing(run_dir, series, _config())
    assert not result.passed
    assert any("never left its initialisation" in f for f in result.failures)


def test_routing_check_is_skipped_when_routing_is_off(tmp_path):
    run_dir = _healthy_run(tmp_path, budget=0.0, keep_end=0.982)
    series = parse_metric_series(run_dir)
    assert check_routing(run_dir, series, _config(mode="none")).passed


def test_routing_check_rejects_a_run_with_no_resolved_config(tmp_path):
    """A job that died before Hydra wrote config.yaml must not pass silently.

    With an empty config the router mode would read as the "none" default, and
    the check would report "routing disabled by config; nothing to verify" for a
    run that never started -- which is how one run first
    looked.
    """
    run_dir = _healthy_run(tmp_path)
    result = check_routing(run_dir, parse_metric_series(run_dir), {})
    assert not result.passed
    assert any("no resolved config.yaml" in f for f in result.failures)


def test_training_check_rejects_a_rising_loss(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "weights" / "step_000100.pt").write_bytes(b"x")
    (run_dir / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _write_log(run_dir, [f"step {i} loss_action={0.5 + 0.02 * i:.4f}" for i in range(30)])
    result = check_training(run_dir, parse_metric_series(run_dir))
    assert not result.passed
    assert any("did not decrease" in f for f in result.failures)


def test_training_check_rejects_a_missing_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _write_log(run_dir, [f"step {i} loss_action={1.0 - 0.01 * i:.4f}" for i in range(30)])
    result = check_training(run_dir, parse_metric_series(run_dir))
    assert not result.passed
    assert any("no checkpoint" in f for f in result.failures)


def _phase_summary(tmp_path, *, episodes=12, semantic=-0.3, geometric=0.4, verdict=True, holes=0):
    bins = 10
    row = [None if i < holes else 0.5 for i in range(bins)]
    payload = {
        "aggregate": {
            "groups": ["dino@t0/primary", "depth@t0/wrist"],
            "num_bins": bins,
            "num_episodes": episodes,
            "keep_ratio": [row, list(row)],
        },
        "stats": {
            "per_curve": {
                "semantic/primary": {"early": 0.8, "late": 0.8 + semantic, "delta": semantic},
                "geometric/wrist": {"early": 0.2, "late": 0.2 + geometric, "delta": geometric},
            },
            "hypothesis_supported": verdict,
        },
    }
    path = tmp_path / "phase_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_visualisation_check_passes_on_a_well_populated_figure(tmp_path):
    assert check_visualisation(_phase_summary(tmp_path)).passed


def test_visualisation_check_rejects_too_few_episodes(tmp_path):
    result = check_visualisation(_phase_summary(tmp_path, episodes=3))
    assert not result.passed
    assert any("too few for a phase curve" in f for f in result.failures)


def test_visualisation_check_rejects_a_mostly_empty_figure(tmp_path):
    result = check_visualisation(_phase_summary(tmp_path, holes=6))
    assert not result.passed
    assert any("mostly holes" in f for f in result.failures)


def test_visualisation_check_recomputes_the_verdict(tmp_path):
    """A stored verdict that does not follow from the stored deltas is a bug."""
    result = check_visualisation(
        _phase_summary(tmp_path, semantic=+0.3, geometric=+0.4, verdict=True)
    )
    assert not result.passed
    assert any("disagrees with the stored deltas" in f for f in result.failures)


def test_visualisation_check_requires_a_summary(tmp_path):
    result = check_visualisation(tmp_path / "missing.json")
    assert not result.passed
    assert any("no phase_summary.json" in f for f in result.failures)


def test_zero_dream_loss_is_rejected(tmp_path):
    """A dream loss pinned at exactly 0 means the extras never loaded.

    Observed on an internal run: the run trains, loss_action falls,
    checkpoints appear -- and the dream branch learns nothing because every
    future is masked invalid. Nothing else in the pipeline notices.
    """
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "weights" / "step_001000.pt").write_bytes(b"x")
    (run_dir / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _write_log(run_dir, [
        f"step {i} loss_action={1.0 - 0.01 * i:.4f} loss_dream=0.0000" for i in range(30)
    ])
    result = check_training(run_dir, parse_metric_series(run_dir), _config())
    assert not result.passed
    assert any("exactly 0.0000" in f for f in result.failures)


def test_gate_movement_is_judged_on_gate_mean_not_keep_ratio(tmp_path):
    """keep_ratio is 1.0 at init by construction; only the gate moves early.

    `keep` is `gate > gate_threshold` (1e-3), so a healthy early run logs
    router_gate_mean=0.998 alongside router_keep_ratio=1.0000 (observed on job
    ). Judging on keep_ratio would call that run dead.
    """
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "weights" / "step_001000.pt").write_bytes(b"x")
    (run_dir / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _write_log(run_dir, [
        f"step {i} loss_router_budget=0.005 router_keep_ratio=1.0000 "
        f"router_gate_mean={0.982 - 0.02 * i:.4f} "
        f"router_keep_dino@t0/primary=1.0000 router_keep_depth@t0/wrist=0.9000"
        for i in range(30)
    ])
    result = check_routing(run_dir, parse_metric_series(run_dir), _config())
    assert result.passed, result.failures


def _ckpt_with_out_scale(run_dir: Path, scale: float) -> None:
    import torch

    (run_dir / "checkpoints" / "weights").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mot": {
                "mixtures.dream.target_encoders.dino.out_scale": torch.tensor([scale]),
                "mixtures.dream.target_encoders.depth.out_scale": torch.tensor([scale]),
            },
            "step": 416,
        },
        # Same name the healthy fixture writes: verify_run reads the *latest*
        # checkpoint, and a leftover dummy sorting after this one would be
        # picked instead and silently yield no out_scale at all.
        run_dir / "checkpoints" / "weights" / "step_001000.pt",
    )


def _generative_config(enabled=True):
    cfg = _config()
    cfg["model"]["generative_dream"] = {"enabled": enabled}
    return cfg


def test_stalled_target_encoder_scale_is_rejected(tmp_path):
    """out_scale still ~0 means the generative Dream never saw its own input.

    Measured once: out_scale reached 0.012/0.018 after 416 steps
    and loss_dream sat at the conditional-mean baseline (~1.21) -- a loss that
    "converged" to something unlearnable. Nothing else in the stack noticed.
    """
    run_dir = _healthy_run(tmp_path)
    _ckpt_with_out_scale(run_dir, 0.012)
    result = check_training(run_dir, parse_metric_series(run_dir), _generative_config())
    assert not result.passed
    assert any("out_scale is still ~0" in f for f in result.failures)


def test_healthy_target_encoder_scale_passes(tmp_path):
    run_dir = _healthy_run(tmp_path)
    _ckpt_with_out_scale(run_dir, 0.93)
    assert check_training(run_dir, parse_metric_series(run_dir), _generative_config()).passed


def test_out_scale_is_not_checked_for_a_regression_dream(tmp_path):
    """A regression Dream has no target encoder, so the check must not fire."""
    run_dir = _healthy_run(tmp_path)
    _ckpt_with_out_scale(run_dir, 0.0)
    assert check_training(
        run_dir, parse_metric_series(run_dir), _generative_config(enabled=False)
    ).passed
