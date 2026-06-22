#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.models.wan22.action_stats import sample_action_noise_like


def _shape(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    return None


def _pick_action_stats(stats, action_key: str | None):
    action = stats.get("action")
    if not isinstance(action, dict) or not action:
        raise ValueError("dataset stats do not contain a non-empty 'action' section.")
    if action_key is None:
        action_key = next(iter(action.keys()))
    if action_key not in action:
        raise ValueError(f"action key {action_key!r} not found. Available keys: {list(action.keys())}")
    return action_key, action[action_key]


def _normalize(action, action_stats, use_per_timestep: bool):
    if use_per_timestep and "action_mean_timestep" in action_stats and "action_std_timestep" in action_stats:
        mean = action_stats["action_mean_timestep"][: action.shape[-2], : action.shape[-1]]
        std = action_stats["action_std_timestep"][: action.shape[-2], : action.shape[-1]].clamp_min(1e-6)
    elif use_per_timestep and "stepwise_mean" in action_stats and "stepwise_std" in action_stats:
        mean = action_stats["stepwise_mean"][: action.shape[-2], : action.shape[-1]]
        std = action_stats["stepwise_std"][: action.shape[-2], : action.shape[-1]].clamp_min(1e-6)
    else:
        mean = action_stats["global_mean"][: action.shape[-1]]
        std = action_stats["global_std"][: action.shape[-1]].clamp_min(1e-6)
    return (action - mean.unsqueeze(0)) / std.unsqueeze(0), mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect FastWAM action stats and correlated action noise.")
    parser.add_argument("dataset_stats_path", type=Path)
    parser.add_argument("--action-key", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--use-per-timestep-norm", action="store_true")
    args = parser.parse_args()

    stats = load_dataset_stats_from_json(str(args.dataset_stats_path))
    action_key, action_stats = _pick_action_stats(stats, args.action_key)
    print(f"dataset_stats_path: {args.dataset_stats_path}")
    print(f"action key: {action_key}")
    print(f"global_mean shape: {_shape(action_stats.get('global_mean'))}")
    print(f"global_std shape: {_shape(action_stats.get('global_std'))}")
    print(f"stepwise_mean shape: {_shape(action_stats.get('stepwise_mean'))}")
    print(f"stepwise_std shape: {_shape(action_stats.get('stepwise_std'))}")
    print(f"action_mean_timestep shape: {_shape(action_stats.get('action_mean_timestep'))}")
    print(f"action_std_timestep shape: {_shape(action_stats.get('action_std_timestep'))}")
    print(f"action_correlation_cholesky shape: {_shape(action_stats.get('action_correlation_cholesky'))}")

    horizon = int(action_stats.get("action_horizon", action_stats["stepwise_mean"].shape[0]))
    action_dim = int(action_stats.get("action_dim", action_stats["global_mean"].shape[0]))
    action = torch.randn(args.batch_size, horizon, action_dim)
    action_norm, mean, std = _normalize(action, action_stats, args.use_per_timestep_norm)
    action_roundtrip = action_norm * std.unsqueeze(0) + mean.unsqueeze(0)
    print(f"normalize -> unnormalize max error: {(action - action_roundtrip).abs().max().item():.6g}")

    chol = action_stats.get("action_correlation_cholesky")
    if chol is None:
        print("action_correlation_cholesky missing: correlated noise sampling cannot be tested.")
        return
    noise = sample_action_noise_like(
        action,
        use_correlated_noise=True,
        action_correlation_cholesky=chol,
    )
    flat = noise.reshape(noise.shape[0], -1).to(torch.float32)
    diag = flat.var(dim=0, unbiased=True)
    print(f"noise shape: {tuple(noise.shape)}")
    print(f"noise mean/std: {noise.mean().item():.6g} / {noise.std(unbiased=True).item():.6g}")
    print(f"empirical covariance diag min/max: {diag.min().item():.6g} / {diag.max().item():.6g}")


if __name__ == "__main__":
    main()
