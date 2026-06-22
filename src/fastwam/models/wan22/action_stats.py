from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch


def _extract_single_action_stats(stats: dict[str, Any], action_key: str = "default") -> dict[str, Any]:
    if "action" not in stats:
        raise ValueError("dataset stats must contain an 'action' section.")
    action_stats = stats["action"]
    if action_key in action_stats:
        return action_stats[action_key]
    if len(action_stats) == 1:
        return next(iter(action_stats.values()))
    raise ValueError(
        f"dataset stats action key {action_key!r} not found and multiple action keys exist: {list(action_stats.keys())}"
    )


def load_action_correlation_cholesky(
    dataset_stats_path: str | Path,
    action_key: str = "default",
) -> torch.Tensor:
    path = Path(dataset_stats_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"dataset stats file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    action_stats = _extract_single_action_stats(stats, action_key=action_key)
    if "action_correlation_cholesky" not in action_stats:
        raise ValueError(
            "Correlated action noise is enabled but dataset stats do not contain "
            f"'action.{action_key}.action_correlation_cholesky'. Regenerate dataset_stats.json with the new stats code."
        )
    return torch.as_tensor(action_stats["action_correlation_cholesky"], dtype=torch.float32)


def validate_action_correlation_cholesky(chol: torch.Tensor, horizon: int, action_dim: int) -> None:
    expected = int(horizon) * int(action_dim)
    got = list(chol.shape)
    if chol.ndim != 2 or chol.shape[0] != expected or chol.shape[1] != expected:
        raise ValueError(
            "action_correlation_cholesky shape mismatch: "
            f"expected [{expected}, {expected}] for H={horizon}, D={action_dim}, got {got}"
        )


def sample_action_noise_like(
    action: torch.Tensor,
    *,
    use_correlated_noise: bool = False,
    action_correlation_cholesky: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if not use_correlated_noise:
        return torch.randn(
            action.shape,
            generator=generator,
            device=action.device,
            dtype=action.dtype,
        )
    if action.ndim != 3:
        raise ValueError(f"Correlated action noise expects action shape [B,H,D], got {tuple(action.shape)}")
    if action_correlation_cholesky is None:
        raise ValueError("Correlated action noise is enabled but action_correlation_cholesky is not loaded.")

    batch_size, horizon, action_dim = action.shape
    validate_action_correlation_cholesky(action_correlation_cholesky, horizon=horizon, action_dim=action_dim)

    chol = action_correlation_cholesky.to(device=action.device, dtype=torch.float32)
    eps = torch.randn(
        (batch_size, horizon * action_dim),
        generator=generator,
        device=action.device,
        dtype=torch.float32,
    )
    noise_flat = eps @ chol.T
    return noise_flat.reshape(batch_size, horizon, action_dim).to(dtype=action.dtype)
