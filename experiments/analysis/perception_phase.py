"""Stage-dependent perception: what does the policy look at, and when?

The hypothesis under test is that a routed policy does not use one fixed view of
the world for a whole episode. Early on, while it is still deciding *what* to do,
it should lean on global semantics -- DINO/SAM over the third-person camera.
As the gripper approaches contact, the decision becomes *where exactly*, which
is local geometry, and the wrist camera is the only view with the resolution to
supply it. If that is true, the router's gate over the Dream tokens should shift
from `dino|sam @ primary` towards `depth @ wrist` as the episode progresses.

This module separates the two halves of that measurement:

* the **core** -- collection, binning and plotting -- is pure and unit-tested,
  so the figure cannot be produced by accident from malformed records;
* the **driver** wraps `experiments/libero/eval_libero_single.py` by
  monkey-patching, the same technique `eval_libero_threshold_single.py` uses, so
  no shipped evaluation code has to change.

Requires `router.group_granularity` to include `_camera`; without the camera
split the primary/wrist question is unanswerable and the script refuses to run.

    python experiments/analysis/perception_phase.py \
      task=routed_wam_libero_goal ckpt=<path> gpu_id=0 \
      EVALUATION.task_suite_name=libero_goal EVALUATION.task_id=0 \
      EVALUATION.num_trials=20 EVALUATION.output_dir=<dir>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: Which perceptual family each Dream modality belongs to. `dyn` is a motion
#: cue rather than geometry or semantics, so it is kept separate instead of
#: being folded into either side of the hypothesis.
MODALITY_FAMILY = {
    "dino": "semantic",
    "sam": "semantic",
    "depth": "geometric",
    "dyn": "motion",
}


def split_group_name(name: str) -> dict[str, Optional[str]]:
    """`depth@t1/wrist` -> {modality, horizon, camera}."""
    camera = None
    body = name
    if "/" in body:
        body, camera = body.split("/", 1)
    horizon = None
    if "@" in body:
        body, horizon = body.split("@", 1)
    return {
        "modality": body,
        "horizon": horizon,
        "camera": camera,
        "family": MODALITY_FAMILY.get(body, "other"),
    }


@dataclass
class PhaseCollector:
    """Accumulate per-replan router group statistics, tagged by episode phase.

    One record per (episode, replan). Progress can only be normalised once the
    episode ends -- an episode that succeeds early is shorter -- so the
    normalisation happens in :meth:`finish_episode`, not at record time.
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    _episode: Optional[int] = None
    _pending: list[dict[str, Any]] = field(default_factory=list)

    def begin_episode(self, episode_idx: int) -> None:
        self._episode = int(episode_idx)
        self._pending = []

    def record(self, router_statistics: Iterable[dict[str, Any]], *, env_step: int) -> None:
        """Snapshot one replan's per-layer, per-group keep ratios."""
        if self._episode is None:
            return
        per_group: dict[str, list[float]] = defaultdict(list)
        layers = 0
        for layer_record in router_statistics:
            layers += 1
            for group, value in layer_record.get("per_group_keep_ratio", {}).items():
                per_group[group].append(float(value))
        if not per_group:
            return
        self._pending.append(
            {
                "episode": self._episode,
                "replan": len(self._pending),
                "env_step": int(env_step),
                "layers": layers,
                "keep_ratio": {g: sum(v) / len(v) for g, v in per_group.items()},
            }
        )

    def finish_episode(self, *, success: bool) -> int:
        if self._episode is None or not self._pending:
            self._episode = None
            self._pending = []
            return 0
        total = len(self._pending)
        denominator = max(total - 1, 1)
        for entry in self._pending:
            entry["progress"] = entry["replan"] / denominator
            entry["success"] = bool(success)
            self.records.append(entry)
        self._episode = None
        self._pending = []
        return total

    def abort_episode(self) -> None:
        self._episode = None
        self._pending = []

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"records": self.records}, indent=2), encoding="utf-8")
        return path


def aggregate_by_phase(
    records: list[dict[str, Any]],
    *,
    num_bins: int = 10,
    successful_only: bool = False,
) -> dict[str, Any]:
    """Mean keep ratio per (group, progress bin).

    Binning is on *normalised* progress, so episodes of different lengths
    contribute equally rather than long episodes dominating the late bins.
    """
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}.")
    rows = [r for r in records if not successful_only or r.get("success")]
    if not rows:
        raise ValueError("No records to aggregate (is successful_only too strict?).")

    groups = sorted({g for r in rows for g in r["keep_ratio"]})
    sums: dict[tuple[str, int], float] = defaultdict(float)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        index = min(int(row["progress"] * num_bins), num_bins - 1)
        for group, value in row["keep_ratio"].items():
            sums[(group, index)] += float(value)
            counts[(group, index)] += 1

    matrix = [
        [
            (sums[(g, b)] / counts[(g, b)]) if counts[(g, b)] else None
            for b in range(num_bins)
        ]
        for g in groups
    ]
    return {
        "groups": groups,
        "num_bins": num_bins,
        "bin_centers": [(b + 0.5) / num_bins for b in range(num_bins)],
        "keep_ratio": matrix,
        "num_records": len(rows),
        "num_episodes": len({r["episode"] for r in rows}),
    }


def family_camera_trends(aggregate: dict[str, Any]) -> dict[str, list[Optional[float]]]:
    """Collapse groups into `family/camera` curves, e.g. `geometric/wrist`."""
    buckets: dict[str, list[list[float]]] = defaultdict(
        lambda: [[] for _ in range(aggregate["num_bins"])]
    )
    for group, row in zip(aggregate["groups"], aggregate["keep_ratio"]):
        parts = split_group_name(group)
        camera = parts["camera"] or "all"
        key = f"{parts['family']}/{camera}"
        for index, value in enumerate(row):
            if value is not None:
                buckets[key][index].append(value)
    return {
        key: [sum(v) / len(v) if v else None for v in bins]
        for key, bins in sorted(buckets.items())
    }


def shift_statistics(trends: dict[str, list[Optional[float]]]) -> dict[str, Any]:
    """Quantify the claimed early-to-late shift as a single comparable number.

    For each curve: mean over the first third of the episode, mean over the last
    third, and their difference. The hypothesis predicts a negative delta for
    `semantic/primary` and a positive delta for `geometric/wrist`.
    """
    out: dict[str, Any] = {}
    for key, values in trends.items():
        present = [(i, v) for i, v in enumerate(values) if v is not None]
        if not present:
            continue
        n = len(values)
        early = [v for i, v in present if i < max(n // 3, 1)]
        late = [v for i, v in present if i >= n - max(n // 3, 1)]
        if not early or not late:
            continue
        early_mean = sum(early) / len(early)
        late_mean = sum(late) / len(late)
        out[key] = {
            "early": early_mean,
            "late": late_mean,
            "delta": late_mean - early_mean,
        }
    verdict = None
    if "semantic/primary" in out and "geometric/wrist" in out:
        verdict = bool(
            out["semantic/primary"]["delta"] < 0.0 and out["geometric/wrist"]["delta"] > 0.0
        )
    return {"per_curve": out, "hypothesis_supported": verdict}


def plot(aggregate: dict[str, Any], trends: dict[str, list[Optional[float]]], out_path: str | Path):
    """Heatmap of every group plus the collapsed family/camera curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.array(
        [[np.nan if v is None else v for v in row] for row in aggregate["keep_ratio"]],
        dtype=float,
    )
    height = max(3.0, 0.32 * len(aggregate["groups"]) + 1.5)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(9.0, height + 3.4), height_ratios=[height, 3.2]
    )

    image = ax_top.imshow(data, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax_top.set_yticks(range(len(aggregate["groups"])))
    ax_top.set_yticklabels(aggregate["groups"], fontsize=8)
    ax_top.set_xticks(range(aggregate["num_bins"]))
    ax_top.set_xticklabels([f"{c:.2f}" for c in aggregate["bin_centers"]], fontsize=8)
    ax_top.set_xlabel("normalised episode progress")
    ax_top.set_title(
        f"Router keep ratio by Dream group and episode phase "
        f"({aggregate['num_episodes']} episodes)"
    )
    fig.colorbar(image, ax=ax_top, label="keep ratio")

    for key, values in trends.items():
        xs = [c for c, v in zip(aggregate["bin_centers"], values) if v is not None]
        ys = [v for v in values if v is not None]
        if not xs:
            continue
        style = "-" if key.endswith("/wrist") else "--"
        ax_bottom.plot(xs, ys, style, marker="o", markersize=3, label=key)
    ax_bottom.set_xlabel("normalised episode progress")
    ax_bottom.set_ylabel("keep ratio")
    ax_bottom.set_title("Collapsed by perceptual family and camera (solid = wrist)")
    ax_bottom.legend(fontsize=7, ncol=2)
    ax_bottom.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def format_report(stats: dict[str, Any]) -> str:
    lines = [f"{'curve':>22} {'early':>8} {'late':>8} {'delta':>8}"]
    for key, value in sorted(stats["per_curve"].items()):
        lines.append(
            f"{key:>22} {value['early']:>8.3f} {value['late']:>8.3f} {value['delta']:>+8.3f}"
        )
    lines.append("")
    verdict = stats["hypothesis_supported"]
    if verdict is None:
        lines.append(
            "Hypothesis not evaluable: need both `semantic/primary` and "
            "`geometric/wrist` curves, i.e. a `_camera` group granularity and "
            "both DINO/SAM and depth modalities enabled."
        )
    elif verdict:
        lines.append(
            "Hypothesis SUPPORTED: semantic/primary falls and geometric/wrist "
            "rises across the episode."
        )
    else:
        lines.append(
            "Hypothesis NOT supported by these rollouts. Report it as measured; "
            "a flat router is itself a finding about how much the policy adapts."
        )
    return "\n".join(lines)


# ----------------------------------------------------------------- driver
def main() -> int:
    import numpy as np  # noqa: F401  (imported early so failures surface before the sim)

    from experiments.libero import eval_libero_single as base_eval

    collector = PhaseCollector()
    original_episode = base_eval.run_single_episode
    original_chunk = base_eval._predict_action_chunk
    state = {"model": None, "step": 0}

    def tracked_predict_action_chunk(obs, task_description, model, processor, cfg, **kwargs):
        result = original_chunk(obs, task_description, model, processor, cfg, **kwargs)
        router = getattr(getattr(model, "mot", None), "router", None)
        if router is not None:
            collector.record(router.last_statistics, env_step=state["step"])
            state["step"] += 1
        state["model"] = model
        return result

    def tracked_run_single_episode(*args, **kwargs):
        episode_idx = int(kwargs.get("episode_idx", args[6] if len(args) > 6 else 0))
        collector.begin_episode(episode_idx)
        state["step"] = 0
        try:
            result = original_episode(*args, **kwargs)
        except BaseException:
            collector.abort_episode()
            raise
        collector.finish_episode(success=bool(result[0]))
        return result

    base_eval.run_single_episode = tracked_run_single_episode
    base_eval._predict_action_chunk = tracked_predict_action_chunk
    try:
        base_eval.eval_single_process()
    finally:
        base_eval.run_single_episode = original_episode
        base_eval._predict_action_chunk = original_chunk

    if not collector.records:
        print(
            "[perception-phase] no router statistics were captured. Check that "
            "router.mode is not 'none' and that router.log_statistics is true.",
            file=sys.stderr,
        )
        return 1

    out_dir = _REPO_ROOT / "evaluate_results" / "perception_phase"
    out_dir.mkdir(parents=True, exist_ok=True)
    collector.to_json(out_dir / "records.json")
    aggregate = aggregate_by_phase(collector.records)
    trends = family_camera_trends(aggregate)
    stats = shift_statistics(trends)
    (out_dir / "phase_summary.json").write_text(
        json.dumps({"aggregate": aggregate, "trends": trends, "stats": stats}, indent=2),
        encoding="utf-8",
    )
    figure = plot(aggregate, trends, out_dir / "perception_phase.png")
    print(format_report(stats))
    print(f"\nrecords  -> {out_dir / 'records.json'}")
    print(f"figure   -> {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
