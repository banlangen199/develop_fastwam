"""Tests for the stage-dependent perception analysis.

No simulator and no checkpoint: the collector is fed synthetic router statistics
with a known shape, so the aggregation and the verdict logic are pinned
independently of whether any real model exhibits the effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.analysis.perception_phase import (  # noqa: E402
    PhaseCollector,
    aggregate_by_phase,
    family_camera_trends,
    format_report,
    shift_statistics,
    split_group_name,
)


def test_split_group_name_parses_every_granularity():
    assert split_group_name("all") == {
        "modality": "all", "horizon": None, "camera": None, "family": "other"
    }
    assert split_group_name("depth") == {
        "modality": "depth", "horizon": None, "camera": None, "family": "geometric"
    }
    assert split_group_name("dino@t1") == {
        "modality": "dino", "horizon": "t1", "camera": None, "family": "semantic"
    }
    assert split_group_name("depth@t0/wrist") == {
        "modality": "depth", "horizon": "t0", "camera": "wrist", "family": "geometric"
    }
    assert split_group_name("sam/primary")["family"] == "semantic"
    assert split_group_name("dyn/wrist")["family"] == "motion"


def _stats(keep_by_group: dict[str, float], layers: int = 3):
    """Fake `router.last_statistics`: one entry per layer."""
    return [
        {"layer": i, "per_group_keep_ratio": dict(keep_by_group)} for i in range(layers)
    ]


def _shifting_episode(collector: PhaseCollector, episode: int, steps: int, success: bool):
    """Semantics decay, wrist geometry grows -- the hypothesised pattern."""
    collector.begin_episode(episode)
    for step in range(steps):
        phase = step / max(steps - 1, 1)
        collector.record(
            _stats(
                {
                    "dino@t0/primary": 0.9 - 0.6 * phase,
                    "depth@t0/wrist": 0.2 + 0.6 * phase,
                    "depth@t0/primary": 0.4,
                }
            ),
            env_step=step,
        )
    return collector.finish_episode(success=success)


def test_collector_normalises_progress_per_episode():
    collector = PhaseCollector()
    assert _shifting_episode(collector, 0, steps=5, success=True) == 5
    # A shorter episode must still span progress 0..1, so episodes of different
    # lengths contribute equally to the phase bins.
    assert _shifting_episode(collector, 1, steps=3, success=False) == 3

    first = [r for r in collector.records if r["episode"] == 0]
    second = [r for r in collector.records if r["episode"] == 1]
    assert [r["progress"] for r in first] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert [r["progress"] for r in second] == [0.0, 0.5, 1.0]
    assert all(r["layers"] == 3 for r in collector.records)


def test_collector_ignores_records_outside_an_episode_and_on_abort():
    collector = PhaseCollector()
    collector.record(_stats({"depth/wrist": 1.0}), env_step=0)
    assert collector.records == []

    collector.begin_episode(0)
    collector.record(_stats({"depth/wrist": 1.0}), env_step=0)
    collector.abort_episode()
    assert collector.records == []
    assert collector.finish_episode(success=True) == 0


def test_aggregate_bins_on_normalised_progress():
    collector = PhaseCollector()
    _shifting_episode(collector, 0, steps=5, success=True)
    aggregate = aggregate_by_phase(collector.records, num_bins=5)

    assert aggregate["groups"] == ["depth@t0/primary", "depth@t0/wrist", "dino@t0/primary"]
    assert aggregate["num_episodes"] == 1
    wrist = aggregate["keep_ratio"][aggregate["groups"].index("depth@t0/wrist")]
    semantic = aggregate["keep_ratio"][aggregate["groups"].index("dino@t0/primary")]
    present_wrist = [v for v in wrist if v is not None]
    present_semantic = [v for v in semantic if v is not None]
    assert present_wrist[0] < present_wrist[-1], "wrist geometry should rise"
    assert present_semantic[0] > present_semantic[-1], "primary semantics should fall"


def test_successful_only_filter():
    collector = PhaseCollector()
    _shifting_episode(collector, 0, steps=4, success=True)
    _shifting_episode(collector, 1, steps=4, success=False)
    assert aggregate_by_phase(collector.records, num_bins=4)["num_episodes"] == 2
    assert (
        aggregate_by_phase(collector.records, num_bins=4, successful_only=True)["num_episodes"] == 1
    )


def test_family_camera_trends_collapse_groups():
    collector = PhaseCollector()
    _shifting_episode(collector, 0, steps=6, success=True)
    trends = family_camera_trends(aggregate_by_phase(collector.records, num_bins=6))
    assert set(trends) == {"geometric/primary", "geometric/wrist", "semantic/primary"}


def test_shift_statistics_detects_the_hypothesised_pattern():
    collector = PhaseCollector()
    for episode in range(3):
        _shifting_episode(collector, episode, steps=6, success=True)
    stats = shift_statistics(
        family_camera_trends(aggregate_by_phase(collector.records, num_bins=6))
    )
    assert stats["hypothesis_supported"] is True
    assert stats["per_curve"]["semantic/primary"]["delta"] < 0
    assert stats["per_curve"]["geometric/wrist"]["delta"] > 0
    # A constant group must produce a delta of zero, not noise.
    assert stats["per_curve"]["geometric/primary"]["delta"] == pytest.approx(0.0, abs=1e-9)
    assert "SUPPORTED" in format_report(stats)


def test_shift_statistics_reports_a_flat_router_honestly():
    collector = PhaseCollector()
    collector.begin_episode(0)
    for step in range(6):
        collector.record(
            _stats({"dino@t0/primary": 0.5, "depth@t0/wrist": 0.5}), env_step=step
        )
    collector.finish_episode(success=True)
    stats = shift_statistics(
        family_camera_trends(aggregate_by_phase(collector.records, num_bins=6))
    )
    assert stats["hypothesis_supported"] is False
    assert "NOT supported" in format_report(stats)


def test_hypothesis_is_unevaluable_without_camera_groups():
    """Without a `_camera` granularity the primary/wrist question cannot be asked."""
    collector = PhaseCollector()
    collector.begin_episode(0)
    for step in range(4):
        collector.record(_stats({"dino@t0": 0.8, "depth@t0": 0.3}), env_step=step)
    collector.finish_episode(success=True)
    stats = shift_statistics(
        family_camera_trends(aggregate_by_phase(collector.records, num_bins=4))
    )
    assert stats["hypothesis_supported"] is None
    assert "not evaluable" in format_report(stats)


def test_plot_writes_a_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from experiments.analysis.perception_phase import plot

    collector = PhaseCollector()
    for episode in range(2):
        _shifting_episode(collector, episode, steps=5, success=True)
    aggregate = aggregate_by_phase(collector.records, num_bins=5)
    out = plot(aggregate, family_camera_trends(aggregate), tmp_path / "fig.png")
    assert out.is_file() and out.stat().st_size > 5000
