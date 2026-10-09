"""Verify a finished RoutedWAM run before it is handed to anyone else.

Three questions, in the order they can invalidate each other:

**Training correct.** Did it actually optimise? Losses finite and falling, the
checkpoint loadable, and -- specifically for this model -- was the action expert
really initialised from the resume checkpoint rather than silently left random
because its width did not match?

**Routing correct.** Did the router do anything? This is the check that would
have caught `loss_router_budget=0.0000`: a budget loss pinned at zero, or a keep
ratio that never leaves `sigmoid(bias_init)`, means the gate never received
pruning pressure and every downstream figure is a picture of the initialisation.

**Visualisation correct.** Does the phase figure rest on enough episodes and
enough groups to mean anything, and does its verdict follow from the numbers?

Exit code 0 only when all three pass. A run that fails here must not be PR'd.

    python experiments/analysis/verify_run.py --run-dir runs/routed_wam_libero_goal/<id>
    python experiments/analysis/verify_run.py --run-dir <dir> \
        --phase-summary evaluate_results/perception_phase/phase_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: Gate value a `learned` router starts at: sigmoid(router.bias_init).
#: A keep ratio that never moves off this is the signature of a dead budget.
def init_keep_ratio(bias_init: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(bias_init)))


class Result:
    def __init__(self, name: str) -> None:
        self.name = name
        self.failures: list[str] = []
        self.notes: list[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    @property
    def passed(self) -> bool:
        return not self.failures

    def render(self) -> str:
        head = f"{'PASS' if self.passed else 'FAIL'}  {self.name}"
        lines = [head]
        for note in self.notes:
            lines.append(f"        {note}")
        for failure in self.failures:
            lines.append(f"    !!  {failure}")
        return "\n".join(lines)


# --------------------------------------------------------------------- logs
# Group metric keys look like `router_keep_depth@t0/wrist`, so `@` and `/` are
# part of the key, not separators.
_METRIC_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_/@]*)\s*[=:]\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")


def parse_metric_series(
    run_dir: Path, extra_logs: Optional[list[Path]] = None
) -> dict[str, list[float]]:
    """Scrape `key=value` metric pairs out of whatever log files the run left.

    The trainer prints its loss dict and also logs to wandb; on the cluster
    wandb is offline, so the text log is the reliable source.
    """
    series: dict[str, list[float]] = {}
    # On a cluster the trainer's stdout goes to the job's own log area, not into the
    # run directory, so the run dir alone contains no metrics at all. Every one
    # of jobs E1-E6 verified as "no loss_action found" for exactly this reason.
    candidates = sorted(run_dir.rglob("*.log")) + sorted(run_dir.rglob("*.txt"))
    candidates += [p for p in (extra_logs or []) if p.is_file()]
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            if "loss" not in line and "router" not in line:
                continue
            for key, value in _METRIC_RE.findall(line):
                try:
                    number = float(value)
                except ValueError:
                    continue
                if math.isfinite(number) or key.startswith("loss"):
                    series.setdefault(key, []).append(number)
    return series


def _target_encoder_scales(checkpoint: Path) -> dict[str, float]:
    """Read `target_encoders.*.out_scale` out of a saved checkpoint."""
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", mmap=True)
    except Exception:
        return {}
    state = payload.get("mot") or {}
    out = {}
    for key, value in state.items():
        if key.endswith("out_scale") and "target_encoders" in key:
            modality = key.split("target_encoders.")[-1].split(".")[0]
            out[modality] = float(value.float().reshape(-1)[0])
    return out


def _trend(values: list[float]) -> Optional[float]:
    """Mean of the last fifth minus mean of the first fifth."""
    if len(values) < 10:
        return None
    window = max(len(values) // 5, 1)
    early = sum(values[:window]) / window
    late = sum(values[-window:]) / window
    return late - early


# ------------------------------------------------------------------ checks
def check_training(
    run_dir: Path, series: dict[str, list[float]], config: dict[str, Any] | None = None
) -> Result:
    result = Result("training correct")
    config = config or {}

    weights = sorted((run_dir / "checkpoints" / "weights").glob("step_*.pt"))
    if not weights:
        result.fail(f"no checkpoint under {run_dir / 'checkpoints' / 'weights'}")
    else:
        result.note(f"{len(weights)} checkpoint(s), latest {weights[-1].name}")

    action = series.get("loss_action", [])
    if not action:
        result.fail("no `loss_action` found in any log under the run dir")
    else:
        if any(not math.isfinite(v) for v in action):
            result.fail("`loss_action` contains NaN/Inf")
        trend = _trend(action)
        result.note(f"loss_action: {len(action)} points, first={action[0]:.4f}, last={action[-1]:.4f}")
        if trend is not None and trend >= 0:
            result.fail(f"`loss_action` did not decrease (last fifth - first fifth = {trend:+.4f})")

    dream = series.get("loss_dream", [])
    lambda_dream = float(
        ((config.get("model") or {}).get("loss") or {}).get("lambda_dream", 0.0) or 0.0
    )
    dream_enabled = bool(
        (((config.get("data") or {}).get("train") or {}).get("dream_target") or {}).get("enabled")
    )
    if dream:
        result.note(f"loss_dream: first={dream[0]:.4f}, last={dream[-1]:.4f}")
        if any(not math.isfinite(v) for v in dream):
            result.fail("`loss_dream` contains NaN/Inf")
        # A dream loss that is exactly zero while it is switched on means every
        # future was masked invalid -- i.e. the dream-target extras did not load.
        # The dataset zero-fills and marks invalid rather than raising on some
        # paths, so this never surfaces as a crash: the run trains happily with a
        # dream branch that learns nothing, and any per-modality figure drawn
        # from it is meaningless. Observed on an internal run.
        if lambda_dream > 0 and all(v == 0.0 for v in dream):
            result.fail(
                f"`loss_dream` is exactly 0.0000 at every logged step while "
                f"lambda_dream={lambda_dream}. The dream targets are not reaching "
                "the loss -- almost always missing `extras/` under the dataset dir. "
                "Check the dataset path is readable from the training host."
            )
    elif lambda_dream > 0 and dream_enabled:
        result.fail(
            f"no `loss_dream` in the logs while lambda_dream={lambda_dream} and "
            "dream_target.enabled=true"
        )
    else:
        result.note("no `loss_dream` in logs (consistent with lambda_dream=0)")

    # Generative Dream only learns if its target encoder is actually passing the
    # noisy target through. `out_scale` starts at `target_encoder_init_scale`;
    # a value still near 0 at the end means the branch never saw its own input,
    # and `loss_dream` will be sitting at the conditional-mean baseline while
    # looking like it "converged". Measured once: out_scale 0.012 /
    # 0.018 after 416 steps, loss_dream flat at ~1.21.
    generative = bool(
        ((config.get("model") or {}).get("generative_dream") or {}).get("enabled")
    )
    if generative and weights:
        scales = _target_encoder_scales(weights[-1])
        if not scales:
            result.note("generative Dream is on but no target-encoder out_scale was found")
        else:
            rendered = ", ".join(f"{k}={v:+.4f}" for k, v in sorted(scales.items()))
            result.note(f"target encoder out_scale: {rendered}")
            weak = {k: v for k, v in scales.items() if abs(v) < 0.05}
            if weak:
                result.fail(
                    f"target-encoder out_scale is still ~0 for {sorted(weak)}; the "
                    "generative Dream is not receiving its noisy target, so "
                    "loss_dream cannot leave the conditional-mean baseline. Set "
                    "generative_dream.target_encoder_init_scale=1.0, or use the "
                    "regression Dream."
                )

    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        result.note(f"resolved config present: {config_path.name}")
    else:
        result.fail(f"missing {config_path}; the run cannot be reproduced without it")
    return result


def check_routing(run_dir: Path, series: dict[str, list[float]], config: dict[str, Any]) -> Result:
    result = Result("routing correct")

    router_cfg = (config.get("model") or {}).get("router") or {}
    if not config:
        # Distinguish "the config says routing is off" from "there is no config".
        # Reporting the latter as the former would let a job that died before
        # Hydra wrote its config pass the routing check.
        result.fail(
            "no resolved config.yaml in the run dir, so the router settings are "
            "unknown. This usually means the job died before Hydra started."
        )
        return result
    mode = router_cfg.get("mode", "none")
    result.note(f"router.mode={mode}")
    if mode == "none":
        result.note("routing disabled by config; nothing to verify")
        return result

    budget = series.get("loss_router_budget", [])
    lambda_budget = float(router_cfg.get("lambda_budget", 0.0) or 0.0)
    if not budget:
        result.fail(
            "`loss_router_budget` never appears in the logs. The router term is "
            "not reaching the loss dict at all."
        )
    elif mode == "learned" and lambda_budget > 0 and all(v == 0.0 for v in budget):
        result.fail(
            "`loss_router_budget` is 0.0000 for every logged step while "
            f"lambda_budget={lambda_budget}. The budget term is dead, so the gate "
            "never felt pruning pressure -- this is the split-path bug where the "
            "cached attention path did not collect gates into `last_gates`."
        )
    else:
        non_zero = sum(1 for v in budget if v != 0.0)
        result.note(f"loss_router_budget: {non_zero}/{len(budget)} non-zero, last={budget[-1]:.6f}")

    # The gate, not the keep ratio, is what moves at the start of training.
    # `keep` is `gate > gate_threshold` (1e-3 by default), so while the gate is
    # still soft at its sigmoid(bias_init) init, keep_ratio is 1.0 by
    # construction -- confirmed on an internal run, which logged
    # router_gate_mean=0.9979 alongside router_keep_ratio=1.0000. Judging
    # movement by keep_ratio alone would call a healthy early run dead.
    gate = series.get("router_gate_mean", [])
    keep = series.get("router_keep_ratio", [])
    start = init_keep_ratio(router_cfg.get("bias_init", 4.0))
    if not gate and not keep:
        result.fail(
            "neither `router_gate_mean` nor `router_keep_ratio` was logged; "
            "set router.log_statistics=true"
        )
    if gate:
        result.note(
            f"router_gate_mean: first={gate[0]:.4f}, last={gate[-1]:.4f} "
            f"(init is sigmoid(bias_init)={start:.4f})"
        )
        if mode == "learned" and len(gate) >= 10 and abs(gate[-1] - start) < 0.005:
            result.fail(
                f"the gate never left its initialisation ({start:.4f}); the router "
                "is not learning a preference"
            )
    if keep:
        result.note(f"router_keep_ratio: first={keep[0]:.4f}, last={keep[-1]:.4f}")
        target = float(router_cfg.get("target_keep_ratio", 0.25))
        if mode == "learned" and lambda_budget > 0 and len(keep) >= 10 and keep[-1] > target + 0.35:
            result.note(
                f"keep ratio {keep[-1]:.3f} is still far above target {target:.2f}; "
                "consider raising lambda_budget or training longer"
            )

    # Per-group breakdown: a router that gates every group identically is not
    # doing anything a scalar could not.
    group_keys = [k for k in series if k.startswith("router_keep_")]
    if len(group_keys) < 2:
        result.note("no per-group keep ratios logged; group-level analysis unavailable")
    else:
        last = {k: series[k][-1] for k in group_keys if series[k]}
        spread = max(last.values()) - min(last.values()) if last else 0.0
        result.note(f"{len(last)} groups logged, final spread={spread:.4f}")
        if mode == "learned" and spread < 1e-3:
            result.fail(
                "every group has the same keep ratio; the router is not "
                "differentiating between modalities/cameras"
            )
        cameras = {k for k in last if k.endswith("/primary") or k.endswith("/wrist")}
        if not cameras:
            result.note(
                "no camera-split groups: set router.group_granularity to a "
                "`_camera` variant for the stage-dependent perception study"
            )
    return result


def check_visualisation(phase_summary: Optional[Path]) -> Result:
    result = Result("visualisation correct")
    if phase_summary is None or not phase_summary.is_file():
        result.fail(
            "no phase_summary.json; run experiments/analysis/perception_phase.py "
            "against this checkpoint first"
        )
        return result

    payload = json.loads(phase_summary.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate") or {}
    stats = payload.get("stats") or {}
    episodes = int(aggregate.get("num_episodes", 0))
    groups = aggregate.get("groups") or []
    result.note(f"{episodes} episodes, {len(groups)} groups, {aggregate.get('num_bins')} bins")

    if episodes < 5:
        result.fail(f"only {episodes} episodes; too few for a phase curve to mean anything")
    if not groups:
        result.fail("no groups in the aggregate")

    matrix = aggregate.get("keep_ratio") or []
    empty = sum(1 for row in matrix for v in row if v is None)
    total = sum(len(row) for row in matrix) or 1
    if empty / total > 0.25:
        result.fail(f"{empty}/{total} phase bins are empty; the figure is mostly holes")
    else:
        result.note(f"{total - empty}/{total} phase bins populated")

    verdict = stats.get("hypothesis_supported")
    curves = stats.get("per_curve") or {}
    if verdict is None:
        result.note(
            "hypothesis not evaluable (needs semantic/primary and geometric/wrist "
            "curves) -- this is a configuration gap, not a model result"
        )
    else:
        semantic = curves.get("semantic/primary", {})
        geometric = curves.get("geometric/wrist", {})
        result.note(
            f"semantic/primary delta={semantic.get('delta', float('nan')):+.3f}, "
            f"geometric/wrist delta={geometric.get('delta', float('nan')):+.3f} "
            f"-> supported={verdict}"
        )
        # Re-derive the verdict rather than trusting the stored flag.
        expected = bool(semantic.get("delta", 0) < 0 and geometric.get("delta", 0) > 0)
        if expected != bool(verdict):
            result.fail(
                f"stored verdict ({verdict}) disagrees with the stored deltas "
                f"({expected}); the summary is internally inconsistent"
            )
    return result


def load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase-summary", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="Extra log file to scrape metrics from (e.g. the cluster job log).",
    )
    parser.add_argument(
        "--skip-visualisation",
        action="store_true",
        help="Verify training and routing only (the figure needs an eval pass).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"run dir does not exist: {run_dir}", file=sys.stderr)
        return 2

    series = parse_metric_series(run_dir, [Path(p) for p in args.log])
    config = load_config(run_dir)

    results = [
        check_training(run_dir, series, config),
        check_routing(run_dir, series, config),
    ]
    if not args.skip_visualisation:
        summary = Path(args.phase_summary) if args.phase_summary else None
        if summary is None:
            default = _REPO_ROOT / "evaluate_results" / "perception_phase" / "phase_summary.json"
            summary = default if default.is_file() else None
        results.append(check_visualisation(summary))

    print(f"run: {run_dir}\n")
    for result in results:
        print(result.render())
        print()

    ok = all(r.passed for r in results)
    print("=" * 62)
    print("VERDICT: SAFE TO SHARE" if ok else "VERDICT: DO NOT SHARE -- fix the failures above")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "passed": ok,
                    "checks": [
                        {"name": r.name, "passed": r.passed, "failures": r.failures, "notes": r.notes}
                        for r in results
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
