from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parents[1]
for local_source_root in (ROOT, ROOT / "LIBERO", ROOT / "experiments" / "libero"):
    if str(local_source_root) not in sys.path:
        sys.path.insert(0, str(local_source_root))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fastwam_alpha_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fastwam_alpha_numba")


def compute_dense_record(
    probs: torch.Tensor,
    slices: dict[str, list[int] | tuple[int, int]],
    n_valid: torch.Tensor | float | int,
) -> dict[str, torch.Tensor]:
    """Reduce one layer's full Action mixed-attention probabilities.

    Args:
        probs: Full mixed distribution [B,H,A,S].
        slices: Dynamic Video/Dream/Action key ranges.
        n_valid: Valid key count [B,A], scalar, or broadcastable equivalent.
    """
    if probs.ndim != 4:
        raise ValueError(f"probs must be [B,H,A,S], got {tuple(probs.shape)}")
    required = {"video", "dream", "action"}
    if not required.issubset(slices):
        raise ValueError(f"Missing mixed-attention slices: {sorted(required - set(slices))}")
    probs = probs.float()
    probability_sum_error = (probs.sum(dim=-1) - 1.0).abs().amax()
    if not torch.isfinite(probs).all():
        raise FloatingPointError("Dense attention contains NaN/Inf.")

    def key_slice(name: str) -> slice:
        start, end = slices[name]
        return slice(int(start), int(end))

    video_probs = probs[..., key_slice("video")]
    dream_probs = probs[..., key_slice("dream")]
    action_probs = probs[..., key_slice("action")]
    source_mass = torch.stack(
        [
            video_probs.sum(dim=-1).mean(dim=(1, 2)),
            dream_probs.sum(dim=-1).mean(dim=(1, 2)),
            action_probs.sum(dim=-1).mean(dim=(1, 2)),
        ],
        dim=-1,
    )
    n_valid_tensor = torch.as_tensor(n_valid, device=probs.device, dtype=probs.dtype)
    if n_valid_tensor.ndim == 0:
        n_valid_tensor = n_valid_tensor.expand(probs.shape[0], probs.shape[2])
    elif n_valid_tensor.ndim == 1:
        if n_valid_tensor.shape[0] == probs.shape[2]:
            n_valid_tensor = n_valid_tensor.unsqueeze(0).expand(probs.shape[0], -1)
        elif n_valid_tensor.shape[0] == probs.shape[0]:
            n_valid_tensor = n_valid_tensor.unsqueeze(1).expand(-1, probs.shape[2])
    if tuple(n_valid_tensor.shape) != (probs.shape[0], probs.shape[2]):
        raise ValueError(
            f"n_valid must resolve to {(probs.shape[0], probs.shape[2])}, "
            f"got {tuple(n_valid_tensor.shape)}"
        )
    normalized = dream_probs * n_valid_tensor[:, None, :, None]
    score = normalized.mean(dim=1).amax(dim=1)
    dream_token_mass = dream_probs.mean(dim=(1, 2))
    return {
        "score": score,
        "dream_token_mass": dream_token_mass,
        "source_mass": source_mass,
        "probability_sum_error": probability_sum_error.reshape(1),
    }


def alpha_metrics(
    score: torch.Tensor,
    dream_token_mass: torch.Tensor,
    alpha: float,
) -> dict[str, float]:
    """Compute offline K and original-mass retention for one alpha."""
    if score.shape != dream_token_mass.shape:
        raise ValueError(
            f"score/mass shape mismatch: {tuple(score.shape)} vs {tuple(dream_token_mass.shape)}"
        )
    keep = score.float() >= float(alpha)
    k = keep.sum(dim=-1).float()
    mass = dream_token_mass.float()
    dense_total = mass.sum()
    kept_total = (mass * keep).sum()
    retention = kept_total / dense_total.clamp(min=1e-12)
    return {
        "alpha": float(alpha),
        "mean_k": float(k.mean().item()),
        "std_k": float(k.std(unbiased=False).item()),
        "min_k": float(k.min().item()),
        "max_k": float(k.max().item()),
        "keep_ratio": float(keep.float().mean().item()),
        "k_zero_ratio_diagnostic_only": float((k == 0).float().mean().item()),
        "dense_dream_mass": float(dense_total.item()),
        "kept_dense_dream_mass": float(kept_total.item()),
        "mass_retention": float(retention.item()),
    }


def resolve_diffusion_step_indices(raw: Any, num_inference_steps: int) -> list[int]:
    """Resolve configured capture positions without silently skipping denoising steps."""
    total = int(num_inference_steps)
    if total <= 0:
        raise ValueError("profile.num_inference_steps must be positive.")
    if isinstance(raw, str):
        if raw.strip().lower() != "all":
            raise ValueError("profile.diffusion_step_indices must be 'all' or a list of integers.")
        return list(range(total))
    steps = sorted({int(value) for value in raw})
    if not steps:
        raise ValueError("profile.diffusion_step_indices must not be empty.")
    invalid = [value for value in steps if value < 0 or value >= total]
    if invalid:
        raise ValueError(
            f"diffusion_step_indices must be in [0,{total}), got {invalid}"
        )
    return steps


def first_step_reuse_metrics(
    score: torch.Tensor,
    dream_token_mass: torch.Tensor,
    alpha: float,
    *,
    diffusion_step_indices: Iterable[int] | None = None,
    anchor_step_index: int = 0,
) -> list[dict[str, float | int]]:
    """Evaluate reusing one dense anchor-step mask over a denoising trajectory.

    The time axis is dimension 1 and the Dream-token axis is last. All other
    axes are treated as independent units (normally sample and Action layer).
    Attention mass always comes from the dense trajectory; this function does
    not claim to measure the causal trajectory after pruning is enabled.
    """
    if score.shape != dream_token_mass.shape:
        raise ValueError(
            f"score/mass shape mismatch: {tuple(score.shape)} vs "
            f"{tuple(dream_token_mass.shape)}"
        )
    if score.ndim < 3:
        raise ValueError("reuse analysis expects [sample,time,...,dream_token] tensors.")
    if diffusion_step_indices is None:
        steps = list(range(int(score.shape[1])))
    else:
        steps = [int(value) for value in diffusion_step_indices]
    if len(steps) != int(score.shape[1]) or len(set(steps)) != len(steps):
        raise ValueError("diffusion_step_indices must uniquely match the profile time axis.")
    if int(anchor_step_index) not in steps:
        raise ValueError(f"anchor step {anchor_step_index} is absent from captured steps {steps}.")

    anchor_position = steps.index(int(anchor_step_index))
    score = score.float()
    mass = dream_token_mass.float()
    anchor_keep = score[:, anchor_position] >= float(alpha)
    anchor_k = anchor_keep.sum(dim=-1).float()

    def safe_global_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
        denominator_value = float(denominator.sum().item())
        if denominator_value == 0.0:
            return 1.0
        return float(numerator.sum().item() / denominator_value)

    rows: list[dict[str, float | int]] = []
    for step_position, step_index in enumerate(steps):
        dynamic_keep = score[:, step_position] >= float(alpha)
        dynamic_k = dynamic_keep.sum(dim=-1).float()
        intersection = anchor_keep & dynamic_keep
        union = anchor_keep | dynamic_keep
        new_tokens = dynamic_keep & ~anchor_keep

        intersection_k = intersection.sum(dim=-1).float()
        union_k = union.sum(dim=-1).float()
        per_unit_recall = torch.where(
            dynamic_k > 0,
            intersection_k / dynamic_k.clamp_min(1.0),
            torch.ones_like(dynamic_k),
        )
        per_unit_jaccard = torch.where(
            union_k > 0,
            intersection_k / union_k.clamp_min(1.0),
            torch.ones_like(union_k),
        )
        step_mass = mass[:, step_position]
        dense_mass = step_mass.sum()
        fixed_kept_mass = (step_mass * anchor_keep).sum()
        dynamic_kept_mass = (step_mass * dynamic_keep).sum()
        rows.append(
            {
                "alpha": float(alpha),
                "anchor_diffusion_step_index": int(anchor_step_index),
                "diffusion_step_index": int(step_index),
                "fixed_mean_k": float(anchor_k.mean().item()),
                "fixed_std_k": float(anchor_k.std(unbiased=False).item()),
                "fixed_keep_ratio": float(anchor_keep.float().mean().item()),
                "dynamic_mean_k": float(dynamic_k.mean().item()),
                "dynamic_std_k": float(dynamic_k.std(unbiased=False).item()),
                "dynamic_keep_ratio": float(dynamic_keep.float().mean().item()),
                "mask_jaccard_micro": safe_global_ratio(intersection, union),
                "later_token_recall_micro": safe_global_ratio(intersection, dynamic_keep),
                "anchor_token_precision_micro": safe_global_ratio(intersection, anchor_keep),
                "mean_new_tokens_per_unit": float(new_tokens.sum(dim=-1).float().mean().item()),
                "new_token_ratio_of_dynamic": safe_global_ratio(new_tokens, dynamic_keep),
                "per_unit_recall_q10": float(torch.quantile(per_unit_recall.flatten(), 0.1).item()),
                "per_unit_recall_q50": float(torch.quantile(per_unit_recall.flatten(), 0.5).item()),
                "per_unit_recall_below_0_9_ratio": float(
                    (per_unit_recall < 0.9).float().mean().item()
                ),
                "per_unit_recall_below_0_8_ratio": float(
                    (per_unit_recall < 0.8).float().mean().item()
                ),
                "per_unit_jaccard_q10": float(
                    torch.quantile(per_unit_jaccard.flatten(), 0.1).item()
                ),
                "dense_dream_mass": float(dense_mass.item()),
                "fixed_kept_dense_dream_mass": float(fixed_kept_mass.item()),
                "fixed_mass_retention": float(
                    (fixed_kept_mass / dense_mass.clamp_min(1e-12)).item()
                ),
                "dynamic_kept_dense_dream_mass": float(dynamic_kept_mass.item()),
                "dynamic_mass_retention": float(
                    (dynamic_kept_mass / dense_mass.clamp_min(1e-12)).item()
                ),
            }
        )
    return rows


def choose_first_step_reuse_alpha(
    rows: Iterable[dict[str, Any]],
    paired: dict[float, dict[str, float]],
    *,
    min_later_token_recall: float,
    min_fixed_mass_retention: float,
    max_relative_action_loss_delta: float,
) -> dict[str, Any]:
    """Choose the smallest fixed K satisfying dense-trajectory diagnostics."""
    by_alpha: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_alpha[float(row["alpha"])].append(dict(row))
    candidates = []
    summaries = []
    for alpha, alpha_rows in sorted(by_alpha.items()):
        later_rows = [
            row
            for row in alpha_rows
            if int(row["diffusion_step_index"])
            != int(row["anchor_diffusion_step_index"])
        ]
        evaluated = later_rows or alpha_rows
        loss_delta = paired.get(alpha, {}).get("mean_relative_action_loss_delta")
        summary = {
            "alpha": float(alpha),
            "fixed_mean_k": float(alpha_rows[0]["fixed_mean_k"]),
            "fixed_keep_ratio": float(alpha_rows[0]["fixed_keep_ratio"]),
            "min_later_token_recall": min(
                float(row["later_token_recall_micro"]) for row in evaluated
            ),
            "min_fixed_mass_retention": min(
                float(row["fixed_mass_retention"]) for row in evaluated
            ),
            "min_mask_jaccard": min(float(row["mask_jaccard_micro"]) for row in evaluated),
            "max_mean_new_tokens_per_unit": max(
                float(row["mean_new_tokens_per_unit"]) for row in evaluated
            ),
            "mean_relative_action_loss_delta": (
                None if loss_delta is None else float(loss_delta)
            ),
        }
        summaries.append(summary)
        if (
            alpha > 0.0
            and loss_delta is not None
            and math.isfinite(float(loss_delta))
            and summary["min_later_token_recall"] >= float(min_later_token_recall)
            and summary["min_fixed_mass_retention"] >= float(min_fixed_mass_retention)
            and float(loss_delta) <= float(max_relative_action_loss_delta)
        ):
            candidates.append(summary)
    rule = {
        "min_later_token_recall": float(min_later_token_recall),
        "min_fixed_mass_retention": float(min_fixed_mass_retention),
        "max_relative_action_loss_delta": float(max_relative_action_loss_delta),
        "objective": "min_fixed_mean_k",
        "k_zero_ratio_used_for_selection": False,
        "dense_alpha_zero_eligible": False,
        "action_loss_note": "paired loss uses per-timestep dynamic masks and is a proxy only",
    }
    if not candidates:
        return {
            "selected_alpha": None,
            "status": "no_nonzero_candidate_satisfies_constraints",
            "selection_rule": rule,
            "candidate_summaries": summaries,
        }
    selected = min(
        candidates,
        key=lambda row: (float(row["fixed_mean_k"]), -float(row["alpha"])),
    )
    return {
        "selected_alpha": float(selected["alpha"]),
        "status": "offline_candidate_selected",
        "selection_rule": rule,
        "selected_metrics": selected,
        "candidate_summaries": summaries,
    }


def choose_alpha(
    rows: Iterable[dict[str, Any]],
    *,
    min_mass_retention: float,
    max_relative_action_loss_delta: float,
) -> dict[str, Any]:
    """Choose minimum mean K under mass/loss constraints.

    K=0 frequency is deliberately diagnostic only and never gates selection.
    """
    candidates = []
    for row in rows:
        loss_delta = row.get("mean_relative_action_loss_delta")
        if loss_delta is None or not math.isfinite(float(loss_delta)):
            continue
        if (
            float(row["mass_retention"]) >= float(min_mass_retention)
            and float(loss_delta) <= float(max_relative_action_loss_delta)
        ):
            candidates.append(row)
    if not candidates:
        return {
            "selected_alpha": None,
            "status": "no_candidate_satisfies_constraints",
            "selection_rule": {
                "min_mass_retention": float(min_mass_retention),
                "max_relative_action_loss_delta": float(max_relative_action_loss_delta),
                "objective": "min_mean_k",
                "k_zero_ratio_used_for_selection": False,
            },
        }
    selected = min(candidates, key=lambda row: (float(row["mean_k"]), -float(row["alpha"])))
    return {
        "selected_alpha": float(selected["alpha"]),
        "status": "selected",
        "selection_rule": {
            "min_mass_retention": float(min_mass_retention),
            "max_relative_action_loss_delta": float(max_relative_action_loss_delta),
            "objective": "min_mean_k",
            "k_zero_ratio_used_for_selection": False,
        },
        "selected_metrics": {str(key): _jsonable(value) for key, value in selected.items()},
    }


def build_episode_calibration_split(
    sample_records: Iterable[dict[str, Any]],
    *,
    episode_ratio: float,
    max_episodes_per_task: int | None,
    observations_per_episode: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select complete episodes by task, then uniform observations per episode."""
    if not 0.0 < episode_ratio <= 1.0:
        raise ValueError("episode_ratio must be in (0,1].")
    if observations_per_episode <= 0:
        raise ValueError("observations_per_episode must be positive.")
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in sample_records:
        key = (str(item["task"]), int(item["dataset_index"]), int(item["episode_index"]))
        grouped[key].append(dict(item))
    by_task: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for key in grouped:
        by_task[key[0]].append(key)

    rng = random.Random(int(seed))
    selected: list[dict[str, Any]] = []
    for task in sorted(by_task):
        episodes = sorted(by_task[task], key=lambda key: (key[1], key[2]))
        count = max(1, int(math.ceil(len(episodes) * episode_ratio)))
        if max_episodes_per_task is not None:
            count = min(count, int(max_episodes_per_task))
        chosen = sorted(rng.sample(episodes, k=min(count, len(episodes))), key=lambda key: (key[1], key[2]))
        for key in chosen:
            observations = sorted(grouped[key], key=lambda item: int(item["frame_index"]))
            count_obs = min(observations_per_episode, len(observations))
            positions = np.linspace(0, len(observations) - 1, num=count_obs, dtype=np.int64)
            for position in positions:
                selected.append(dict(observations[int(position)]))
    selected.sort(key=lambda item: (str(item["task"]), int(item["episode_index"]), int(item["frame_index"])))
    return selected


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else (ROOT / path).resolve()


def _load_config(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    required = {"checkpoint", "hydra_task", "output_dir", "calibration", "profile", "alpha_values", "selection"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"Calibration config missing keys: {sorted(missing)}")
    return cfg


def _save_or_validate_resolved_config(cfg: DictConfig, output_dir: Path) -> None:
    path = output_dir / "resolved_config.yaml"
    current = OmegaConf.to_container(cfg, resolve=True)
    if path.exists():
        previous = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if previous != current:
            raise ValueError(
                f"Output directory {output_dir} already belongs to a different configuration. "
                "Use a new output_dir instead of mixing calibration artifacts."
            )
        return
    OmegaConf.save(config=cfg, f=str(path))


def _runtime_imports():
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from torch.utils.data._utils.collate import default_collate

    from experiments.libero.eval_libero_single import (
        _apply_training_model_config,
        _load_model_checkpoint,
        _maybe_load_action_noise_stats,
        _mixed_precision_to_model_dtype,
        _resolve_dataset_stats_path,
    )
    from experiments.libero.evaluate_dream_fixed_k import (
        _action_loss,
        _build_dataset,
        _dream_token_metadata,
        _forward_action,
        _prepare_flow_matching_pair,
    )
    from fastwam.models.wan22.action_dream_threshold import (
        ActionDreamThresholdConfig,
        ThresholdDreamFastWAM,
    )

    return locals()


def _build_runtime(cfg: DictConfig, output_dir: Path):
    api = _runtime_imports()
    checkpoint = _resolve_path(cfg.checkpoint)
    overrides = [
        f"task={cfg.hydra_task}",
        f"ckpt={checkpoint}",
        "EVALUATION.num_trials=1",
    ]
    if cfg.get("dataset_stats_path"):
        overrides.append(f"EVALUATION.dataset_stats_path={_resolve_path(cfg.dataset_stats_path)}")
    with api["initialize_config_dir"](
        config_dir=str((ROOT / "configs").resolve()), version_base="1.3"
    ):
        runtime_cfg = api["compose"](config_name="sim_libero.yaml", overrides=overrides)
    if cfg.get("device"):
        runtime_cfg.EVALUATION.device = str(cfg.device)
    training_config_path = api["_apply_training_model_config"](runtime_cfg)
    dataset_stats_path = api["_resolve_dataset_stats_path"](runtime_cfg)
    runtime_cfg.model.load_text_encoder = False
    device = str(
        cfg.get("device")
        or runtime_cfg.EVALUATION.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_dtype = api["_mixed_precision_to_model_dtype"](
        str(runtime_cfg.get("mixed_precision", "bf16"))
    )
    model = api["instantiate"](runtime_cfg.model, model_dtype=model_dtype, device=device)
    api["_load_model_checkpoint"](model, str(checkpoint))
    api["_maybe_load_action_noise_stats"](model, runtime_cfg, dataset_stats_path)
    model = model.to(device).eval()
    model.requires_grad_(False)
    if not hasattr(model, "dream_expert"):
        raise TypeError("Calibration requires DreamFastWAM.")
    if not hasattr(model.mot, "threshold_config"):
        model = api["ThresholdDreamFastWAM"].from_dense_model(
            model,
            action_dream_threshold={
                "enabled": False,
                "alpha": 1.0,
                "warmup_ratio": 0.0,
                "head_reduce": "mean",
                "action_query_reduce": "max",
                "min_keep_dream_tokens": 0,
                "detach_selection_score": True,
                "log_statistics": False,
                "save_detailed_tensors": False,
            },
            finetune_action_only=True,
        ).eval()
    dataset = api["_build_dataset"](
        runtime_cfg,
        dataset_stats_path=dataset_stats_path,
        output_dir=output_dir,
    )
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "training_config_path": None if training_config_path is None else str(training_config_path),
        "dataset_stats_path": str(dataset_stats_path),
        "device": device,
        "model_dtype": str(model_dtype),
    }
    return runtime_cfg, model, dataset, api, metadata


def _episode_task_maps(runtime_cfg: DictConfig) -> dict[int, dict[int, str]]:
    result: dict[int, dict[int, str]] = {}
    for dataset_index, raw_root in enumerate(runtime_cfg.data.train.dataset_dirs):
        root = _resolve_path(raw_root)
        mapping: dict[int, str] = {}
        with (root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                tasks = item.get("tasks") or ["unknown"]
                mapping[int(item["episode_index"])] = str(tasks[0])
        result[dataset_index] = mapping
    return result


def _all_sample_records(dataset, runtime_cfg: DictConfig) -> list[dict[str, Any]]:
    base = getattr(dataset, "lerobot_dataset", None)
    if base is None or not hasattr(base, "_metadata_from_sample_idx"):
        raise TypeError("Dataset does not expose inexpensive sample metadata.")
    task_maps = _episode_task_maps(runtime_cfg)
    records = []
    for sample_index in range(len(dataset)):
        metadata = base._metadata_from_sample_idx(sample_index)
        dataset_index = int(torch.as_tensor(metadata["dataset_index"]).item())
        episode_index = int(torch.as_tensor(metadata["episode_index"]).item())
        frame_index = int(torch.as_tensor(metadata["frame_index"]).item())
        records.append(
            {
                "sample_index": int(sample_index),
                "dataset_index": dataset_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "task": task_maps.get(dataset_index, {}).get(episode_index, "unknown"),
            }
        )
    return records


def _load_or_create_split(
    cfg: DictConfig,
    runtime_cfg: DictConfig,
    dataset,
    output_dir: Path,
) -> list[dict[str, Any]]:
    split_path = output_dir / "calibration_split.json"
    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        return list(payload["samples"])
    calibration = cfg.calibration
    samples = build_episode_calibration_split(
        _all_sample_records(dataset, runtime_cfg),
        episode_ratio=float(calibration.episode_ratio),
        max_episodes_per_task=(
            None
            if calibration.get("max_episodes_per_task") is None
            else int(calibration.max_episodes_per_task)
        ),
        observations_per_episode=int(calibration.observations_per_episode),
        seed=int(cfg.seed),
    )
    _atomic_json(
        split_path,
        {
            "seed": int(cfg.seed),
            "episode_ratio": float(calibration.episode_ratio),
            "max_episodes_per_task": calibration.get("max_episodes_per_task"),
            "observations_per_episode": int(calibration.observations_per_episode),
            "num_samples": len(samples),
            "samples": samples,
        },
    )
    return samples


def _attention_layers(model, raw: Any) -> list[int]:
    if str(raw).strip().lower() == "all":
        return list(range(int(model.mot.num_layers)))
    layers = [int(value) for value in raw]
    invalid = [value for value in layers if value < 0 or value >= int(model.mot.num_layers)]
    if invalid:
        raise ValueError(f"Invalid attention layers: {invalid}")
    return layers


def _profile_one_sample(
    *,
    model,
    sample: dict[str, Any],
    sample_metadata: dict[str, Any],
    api: dict[str, Any],
    cfg: DictConfig,
    attention_layers: list[int],
    seed: int,
) -> dict[str, Any]:
    _set_seed(seed)
    inputs = model.build_inputs(sample, tiled=bool(cfg.get("tiled", False)))
    first_frame = inputs["first_frame_latents"]
    if first_frame is None:
        first_frame = inputs["input_latents"][:, :, 0:1]
    action_horizon = int(inputs["action"].shape[1])
    latents_action = torch.randn(
        (1, action_horizon, model.action_expert.action_dim),
        device=model.device,
        dtype=model.torch_dtype,
    )
    timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=int(cfg.profile.num_inference_steps),
        device=model.device,
        dtype=latents_action.dtype,
        shift_override=None,
    )
    capture_steps = set(
        resolve_diffusion_step_indices(
            cfg.profile.diffusion_step_indices,
            num_inference_steps=len(timesteps),
        )
    )
    scores = []
    token_masses = []
    source_masses = []
    sum_errors = []
    valid_counts = []
    captured_timesteps = []
    layer_ids = None
    with torch.inference_mode():
        for step_index, (step_t, step_delta) in enumerate(zip(timesteps, deltas)):
            timestep_action = step_t.unsqueeze(0).to(device=model.device, dtype=latents_action.dtype)
            timestep_video = torch.zeros_like(
                timestep_action, dtype=first_frame.dtype, device=model.device
            )
            capture = step_index in capture_steps
            output = model._predict_joint_noise(
                latents_video=first_frame,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=inputs["context"],
                context_mask=inputs["context_mask"],
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                gt_action=None,
                return_action_attention=capture,
                attention_layers=attention_layers,
            )
            pred_action = output[1]
            if capture:
                records = output[2]
                current_scores = []
                current_token_masses = []
                current_source_masses = []
                current_errors = []
                current_valid_counts = []
                current_layers = []
                for record in records:
                    probs = record["probs"].float()
                    n_valid = int(probs.shape[-1])
                    reduced = compute_dense_record(probs, record["slices"], n_valid=n_valid)
                    current_scores.append(reduced["score"][0])
                    current_token_masses.append(reduced["dream_token_mass"][0])
                    current_source_masses.append(reduced["source_mass"][0])
                    current_errors.append(reduced["probability_sum_error"][0])
                    current_valid_counts.append(n_valid)
                    current_layers.append(int(record["layer"]))
                if layer_ids is None:
                    layer_ids = current_layers
                elif layer_ids != current_layers:
                    raise RuntimeError("Captured attention layers changed across diffusion steps.")
                scores.append(torch.stack(current_scores))
                token_masses.append(torch.stack(current_token_masses))
                source_masses.append(torch.stack(current_source_masses))
                sum_errors.append(torch.stack(current_errors))
                valid_counts.append(torch.tensor(current_valid_counts, dtype=torch.int32))
                captured_timesteps.append(float(timestep_action[0].float().item()))
            latents_action = model.infer_action_scheduler.step(
                pred_action, step_delta, latents_action
            )
    return {
        "metadata": dict(sample_metadata),
        "seed": int(seed),
        "diffusion_step_indices": sorted(capture_steps),
        "diffusion_timesteps": torch.tensor(captured_timesteps, dtype=torch.float32),
        "layer_ids": torch.tensor(layer_ids, dtype=torch.int16),
        "score": torch.stack(scores).to(dtype=torch.float16).cpu(),
        "dream_token_mass": torch.stack(token_masses).to(dtype=torch.float16).cpu(),
        "source_mass": torch.stack(source_masses).to(dtype=torch.float32).cpu(),
        "probability_sum_error": torch.stack(sum_errors).to(dtype=torch.float32).cpu(),
        "n_valid": torch.stack(valid_counts).to(dtype=torch.int32).cpu(),
    }


def run_profile(cfg, model, dataset, api, samples, output_dir) -> None:
    shard_dir = output_dir / "profile_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    layers = _attention_layers(model, cfg.profile.attention_layers)
    model.mot.threshold_config = dataclasses.replace(
        model.mot.threshold_config, enabled=False, log_statistics=False
    )
    model.eval()
    token_metadata_path = output_dir / "dream_token_metadata.json"
    if not token_metadata_path.exists():
        slices = model.dream_expert.modality_slices()
        num_dream_tokens = max(int(slc.stop) for slc in slices.values())
        _atomic_json(
            token_metadata_path,
            api["_dream_token_metadata"](model, num_dream_tokens),
        )
    for position, item in enumerate(samples):
        path = shard_dir / f"sample_{position:06d}.pt"
        if path.exists():
            continue
        sample = api["default_collate"]([dataset[int(item["sample_index"])]])
        record = _profile_one_sample(
            model=model,
            sample=sample,
            sample_metadata=item,
            api=api,
            cfg=cfg,
            attention_layers=layers,
            seed=int(cfg.seed) + position * 1009,
        )
        temporary = path.with_suffix(".pt.tmp")
        torch.save(record, temporary)
        temporary.replace(path)
        _atomic_json(
            output_dir / "progress.json",
            {"profile_completed": position + 1, "profile_total": len(samples)},
        )
        print(f"[alpha-profile] {position + 1}/{len(samples)} {item['task']} ep={item['episode_index']}")


def _set_alpha(model, alpha: float) -> None:
    model.mot.threshold_config = dataclasses.replace(
        model.mot.threshold_config,
        enabled=float(alpha) > 0.0,
        alpha=float(alpha),
        warmup_ratio=0.0,
        log_statistics=False,
        save_detailed_tensors=False,
    )
    model.eval()


def run_paired_loss(cfg, model, dataset, api, samples, output_dir) -> None:
    path = output_dir / "paired_action_loss.csv"
    rows: list[dict[str, Any]] = [dict(row) for row in _read_csv(path)]
    row_by_key = {
        (int(row["sample_position"]), int(row["repeat_index"]), float(row["alpha"])): row
        for row in rows
    }
    alpha_values = [float(value) for value in cfg.alpha_values]
    for position, item in enumerate(samples):
        sample = api["default_collate"]([dataset[int(item["sample_index"])]])
        for repeat_index in range(int(cfg.paired_loss.noise_repeats)):
            pair_seed = int(cfg.seed) + position * 1009 + repeat_index
            _set_seed(pair_seed)
            pair = api["_prepare_flow_matching_pair"](
                model, sample, tiled=bool(cfg.get("tiled", False))
            )
            losses: dict[float, float] = {}
            for alpha in alpha_values:
                key = (position, repeat_index, alpha)
                existing = row_by_key.get(key)
                if existing is not None:
                    losses[alpha] = float(existing["action_loss"])
                    continue
                _set_alpha(model, alpha)
                with torch.inference_mode():
                    prediction, _, _ = api["_forward_action"](
                        model=model,
                        pair=pair,
                        selected_dream_indices=None,
                        capture_attention=False,
                        attention_layers=[],
                    )
                    loss = api["_action_loss"](
                        model=model,
                        pred_action=prediction,
                        target_action=pair["target_action"],
                        action_is_pad=pair["action_is_pad"],
                        timestep_action=pair["timestep_action"],
                    )
                losses[alpha] = float(loss.item())
                new_row = {
                        "sample_position": position,
                        "sample_index": int(item["sample_index"]),
                        "task": str(item["task"]),
                        "episode_index": int(item["episode_index"]),
                        "frame_index": int(item["frame_index"]),
                        "repeat_index": repeat_index,
                        "pair_seed": pair_seed,
                        "action_timestep": float(pair["timestep_action"][0].float().item()),
                        "alpha": alpha,
                        "action_loss": losses[alpha],
                    }
                rows.append(new_row)
                row_by_key[key] = new_row
                _write_csv(path, rows)
            dense_loss = losses.get(0.0)
            if dense_loss is None:
                raise ValueError("alpha_values must contain 0.0 as the Dense baseline.")
            for row in rows:
                if int(row["sample_position"]) == position and int(row["repeat_index"]) == repeat_index:
                    loss = float(row["action_loss"])
                    row["dense_action_loss"] = dense_loss
                    row["action_loss_delta"] = loss - dense_loss
                    row["relative_action_loss_delta"] = (loss - dense_loss) / max(abs(dense_loss), 1e-8)
            _write_csv(path, rows)
            print(
                f"[alpha-loss] sample={position + 1}/{len(samples)} repeat={repeat_index + 1} "
                + " ".join(f"a={alpha:g}:{losses[alpha]:.6f}" for alpha in alpha_values)
            )


def _profile_tensors(output_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    paths = sorted((output_dir / "profile_shards").glob("sample_*.pt"))
    if not paths:
        raise FileNotFoundError("No profile shards found. Run --stage profile first.")
    records = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    return (
        torch.stack([record["score"] for record in records]).float(),
        torch.stack([record["dream_token_mass"] for record in records]).float(),
        torch.stack([record["source_mass"] for record in records]).float(),
        records,
    )


def _paired_summary(output_dir: Path) -> dict[float, dict[str, float]]:
    rows = _read_csv(output_dir / "paired_action_loss.csv")
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("relative_action_loss_delta") not in (None, ""):
            grouped[float(row["alpha"])].append(float(row["relative_action_loss_delta"]))
    return {
        alpha: {
            "paired_count": len(values),
            "mean_relative_action_loss_delta": float(np.mean(values)),
            "std_relative_action_loss_delta": float(np.std(values)),
        }
        for alpha, values in grouped.items()
    }


def _save_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[alpha-summary] matplotlib unavailable; skipping plots.")
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    alpha = [float(row["alpha"]) for row in rows]
    fig, left = plt.subplots(figsize=(7, 4.5))
    left.plot(alpha, [float(row["mean_k"]) for row in rows], marker="o", label="mean K")
    left.set_xlabel("alpha")
    left.set_ylabel("mean retained Dream tokens")
    left.grid(alpha=0.25)
    right = left.twinx()
    right.plot(
        alpha,
        [float(row["mass_retention"]) for row in rows],
        marker="s",
        color="tab:orange",
        label="mass retention",
    )
    right.set_ylabel("original Dream attention mass retention")
    fig.tight_layout()
    fig.savefig(plot_dir / "alpha_k_mass.png", dpi=180)
    plt.close(fig)

    if all(row.get("mean_relative_action_loss_delta") is not None for row in rows):
        plt.figure(figsize=(7, 4.5))
        plt.plot(
            alpha,
            [float(row["mean_relative_action_loss_delta"]) for row in rows],
            marker="o",
        )
        plt.axhline(float(0.01), color="tab:red", linestyle="--", linewidth=1)
        plt.xlabel("alpha")
        plt.ylabel("mean relative action-loss delta")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / "alpha_paired_loss.png", dpi=180)
        plt.close()


def _save_first_step_reuse_plots(
    output_dir: Path,
    rows: list[dict[str, Any]],
    report_alphas: Iterable[float],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[alpha-summary] matplotlib unavailable; skipping first-step reuse plots.")
        return
    selected = {float(value) for value in report_alphas}
    available = {float(row["alpha"]) for row in rows}
    selected &= available
    if not selected:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for alpha in sorted(selected):
        alpha_rows = sorted(
            (row for row in rows if float(row["alpha"]) == alpha),
            key=lambda row: int(row["diffusion_step_index"]),
        )
        steps = [int(row["diffusion_step_index"]) for row in alpha_rows]
        axes[0].plot(
            steps,
            [float(row["later_token_recall_micro"]) for row in alpha_rows],
            marker="o",
            label=f"alpha={alpha:g} recall",
        )
        axes[0].plot(
            steps,
            [float(row["mask_jaccard_micro"]) for row in alpha_rows],
            marker="x",
            linestyle="--",
            label=f"alpha={alpha:g} Jaccard",
        )
        axes[1].plot(
            steps,
            [float(row["fixed_mass_retention"]) for row in alpha_rows],
            marker="o",
            label=f"alpha={alpha:g} fixed mask",
        )
        axes[1].plot(
            steps,
            [float(row["dynamic_mass_retention"]) for row in alpha_rows],
            marker="x",
            linestyle="--",
            label=f"alpha={alpha:g} dynamic mask",
        )
    axes[0].set_xlabel("denoising step")
    axes[0].set_ylabel("mask overlap")
    axes[0].set_ylim(0.0, 1.01)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("denoising step")
    axes[1].set_ylabel("dense Dream mass retention")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "first_step_reuse_overlap_mass.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(7, 4.5))
    for alpha in sorted(selected):
        alpha_rows = sorted(
            (row for row in rows if float(row["alpha"]) == alpha),
            key=lambda row: int(row["diffusion_step_index"]),
        )
        steps = [int(row["diffusion_step_index"]) for row in alpha_rows]
        plt.plot(
            steps,
            [float(row["dynamic_mean_k"]) for row in alpha_rows],
            marker="o",
            label=f"alpha={alpha:g} dynamic K",
        )
        plt.plot(
            steps,
            [float(row["fixed_mean_k"]) for row in alpha_rows],
            linestyle="--",
            label=f"alpha={alpha:g} fixed step-0 K",
        )
    plt.xlabel("denoising step")
    plt.ylabel("mean retained Dream tokens")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "first_step_reuse_k.png", dpi=180)
    plt.close()


def run_summary(cfg: DictConfig, output_dir: Path) -> None:
    score, token_mass, source_mass, records = _profile_tensors(output_dir)
    paired = _paired_summary(output_dir)
    step_indices = [int(value) for value in records[0]["diffusion_step_indices"]]
    for record in records[1:]:
        if [int(value) for value in record["diffusion_step_indices"]] != step_indices:
            raise ValueError("Profile shards do not share the same diffusion-step positions.")
    rows = []
    for alpha in [float(value) for value in cfg.alpha_values]:
        row = alpha_metrics(score, token_mass, alpha)
        row.update(paired.get(alpha, {}))
        rows.append(row)
    _write_csv(output_dir / "alpha_sweep.csv", rows)

    dense_summary = {
        "num_profile_samples": int(score.shape[0]),
        "profile_shape": list(score.shape),
        "mean_video_attention_mass": float(source_mass[..., 0].mean().item()),
        "mean_dream_attention_mass": float(source_mass[..., 1].mean().item()),
        "mean_action_self_attention_mass": float(source_mass[..., 2].mean().item()),
        "max_probability_sum_error": float(
            max(record["probability_sum_error"].max().item() for record in records)
        ),
        "score_quantiles": {
            str(q): float(torch.quantile(score.flatten(), q).item())
            for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
    }
    _atomic_json(output_dir / "dense_summary.json", dense_summary)

    per_layer = []
    layer_ids = records[0]["layer_ids"].tolist()
    for layer_position, layer_id in enumerate(layer_ids):
        layer_score = score[:, :, layer_position]
        layer_mass = token_mass[:, :, layer_position]
        layer_source = source_mass[:, :, layer_position]
        base = {
            "layer": int(layer_id),
            "video_attention_mass": float(layer_source[..., 0].mean().item()),
            "dream_attention_mass": float(layer_source[..., 1].mean().item()),
            "action_self_attention_mass": float(layer_source[..., 2].mean().item()),
            "score_mean": float(layer_score.mean().item()),
            "score_q50": float(torch.quantile(layer_score.flatten(), 0.5).item()),
            "score_q90": float(torch.quantile(layer_score.flatten(), 0.9).item()),
        }
        for alpha in [float(value) for value in cfg.alpha_values]:
            metrics = alpha_metrics(layer_score, layer_mass, alpha)
            per_layer.append({**base, **metrics})
    _write_csv(output_dir / "per_layer.csv", per_layer)

    per_timestep = []
    for step_position, step_index in enumerate(step_indices):
        step_score = score[:, step_position]
        step_mass = token_mass[:, step_position]
        step_source = source_mass[:, step_position]
        base = {
            "diffusion_step_index": int(step_index),
            "video_attention_mass": float(step_source[..., 0].mean().item()),
            "dream_attention_mass": float(step_source[..., 1].mean().item()),
            "action_self_attention_mass": float(step_source[..., 2].mean().item()),
        }
        for alpha in [float(value) for value in cfg.alpha_values]:
            per_timestep.append({**base, **alpha_metrics(step_score, step_mass, alpha)})
    _write_csv(output_dir / "per_timestep.csv", per_timestep)

    task_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        task_to_indices[str(record["metadata"]["task"])].append(index)

    reuse_cfg = cfg.get("first_step_reuse_analysis") or {}
    reuse_enabled = bool(reuse_cfg.get("enabled", True))
    if reuse_enabled:
        anchor_step_index = int(reuse_cfg.get("anchor_step_index", 0))
        reuse_rows = []
        for alpha in [float(value) for value in cfg.alpha_values]:
            reuse_rows.extend(
                first_step_reuse_metrics(
                    score,
                    token_mass,
                    alpha,
                    diffusion_step_indices=step_indices,
                    anchor_step_index=anchor_step_index,
                )
            )
        _write_csv(output_dir / "first_step_reuse.csv", reuse_rows)

        reuse_per_layer = []
        for layer_position, layer_id in enumerate(layer_ids):
            for alpha in [float(value) for value in cfg.alpha_values]:
                layer_rows = first_step_reuse_metrics(
                    score[:, :, layer_position],
                    token_mass[:, :, layer_position],
                    alpha,
                    diffusion_step_indices=step_indices,
                    anchor_step_index=anchor_step_index,
                )
                reuse_per_layer.extend(
                    {"layer": int(layer_id), **row} for row in layer_rows
                )
        _write_csv(output_dir / "first_step_reuse_per_layer.csv", reuse_per_layer)

        reuse_per_task = []
        for task, indices in sorted(task_to_indices.items()):
            for alpha in [float(value) for value in cfg.alpha_values]:
                task_rows = first_step_reuse_metrics(
                    score[indices],
                    token_mass[indices],
                    alpha,
                    diffusion_step_indices=step_indices,
                    anchor_step_index=anchor_step_index,
                )
                reuse_per_task.extend(
                    {"task": task, "num_observations": len(indices), **row}
                    for row in task_rows
                )
        _write_csv(output_dir / "first_step_reuse_per_task.csv", reuse_per_task)

        reuse_selection = reuse_cfg.get("selection") or {}
        reuse_recommendation = choose_first_step_reuse_alpha(
            reuse_rows,
            paired,
            min_later_token_recall=float(
                reuse_selection.get("min_later_token_recall", 0.95)
            ),
            min_fixed_mass_retention=float(
                reuse_selection.get("min_fixed_mass_retention", 0.90)
            ),
            max_relative_action_loss_delta=float(
                reuse_selection.get(
                    "max_relative_action_loss_delta",
                    cfg.selection.max_relative_action_loss_delta,
                )
            ),
        )
        reuse_recommendation["analysis_scope"] = {
            "trajectory": "dense_action_denoising",
            "captured_diffusion_step_indices": step_indices,
            "all_denoising_steps_captured": step_indices
            == list(range(int(cfg.profile.num_inference_steps))),
            "anchor_step_index": anchor_step_index,
            "causal_fixed_mask_rollout_measured": False,
            "note": (
                "Later scores and masses are measured on the paired dense trajectory. "
                "A real fixed-mask rollout may change later Action latents and attention."
            ),
        }
        _atomic_json(
            output_dir / "recommended_first_step_alpha.json",
            reuse_recommendation,
        )
        report_alphas = reuse_cfg.get("report_alphas", [0.2, 0.25])
        _save_first_step_reuse_plots(output_dir, reuse_rows, report_alphas)

    per_task = []
    for task, indices in sorted(task_to_indices.items()):
        task_score = score[indices]
        task_mass = token_mass[indices]
        task_source = source_mass[indices]
        base = {
            "task": task,
            "num_observations": len(indices),
            "video_attention_mass": float(task_source[..., 0].mean().item()),
            "dream_attention_mass": float(task_source[..., 1].mean().item()),
            "action_self_attention_mass": float(task_source[..., 2].mean().item()),
        }
        for alpha in [float(value) for value in cfg.alpha_values]:
            per_task.append({**base, **alpha_metrics(task_score, task_mass, alpha)})
    _write_csv(output_dir / "per_task.csv", per_task)

    token_metadata_path = output_dir / "dream_token_metadata.json"
    if token_metadata_path.exists():
        token_metadata = json.loads(token_metadata_path.read_text(encoding="utf-8"))
        token_groups: dict[str, list[int]] = defaultdict(list)
        for item in token_metadata:
            token_groups[f"modality:{item['modality']}"].append(int(item["token_index"]))
            token_groups[f"horizon:{item['future_offset']}"].append(int(item["token_index"]))
        per_token_group = []
        for group, indices in sorted(token_groups.items()):
            group_score = score[..., indices]
            group_mass = token_mass[..., indices]
            for alpha in [float(value) for value in cfg.alpha_values]:
                per_token_group.append(
                    {"token_group": group, **alpha_metrics(group_score, group_mass, alpha)}
                )
        _write_csv(output_dir / "per_token_group.csv", per_token_group)

    recommendation = choose_alpha(
        rows,
        min_mass_retention=float(cfg.selection.min_mass_retention),
        max_relative_action_loss_delta=float(cfg.selection.max_relative_action_loss_delta),
    )
    _atomic_json(output_dir / "recommended_alpha.json", recommendation)
    _save_plots(output_dir, rows)
    print(f"[alpha-summary] Dense Dream mass={dense_summary['mean_dream_attention_mass']:.6f}")
    print(f"[alpha-summary] recommendation={recommendation['selected_alpha']}")
    if reuse_enabled:
        print(
            "[alpha-summary] fixed-step0 offline candidate="
            f"{reuse_recommendation['selected_alpha']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the global Action-to-Dream alpha.")
    parser.add_argument("--config", default="action_dream_alpha/libero_goal.yaml")
    parser.add_argument(
        "--stage",
        choices=("profile", "paired-loss", "summarize", "all"),
        default="all",
    )
    args = parser.parse_args()
    config_path = _resolve_path(args.config)
    cfg = _load_config(config_path)
    output_dir = _resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_or_validate_resolved_config(cfg, output_dir)
    _set_seed(int(cfg.seed))

    if args.stage == "summarize":
        run_summary(cfg, output_dir)
        return

    runtime_cfg, model, dataset, api, metadata = _build_runtime(cfg, output_dir)
    _atomic_json(output_dir / "checkpoint_metadata.json", metadata)
    samples = _load_or_create_split(cfg, runtime_cfg, dataset, output_dir)
    print(f"[alpha] output={output_dir} samples={len(samples)}")

    if args.stage in ("profile", "all"):
        run_profile(cfg, model, dataset, api, samples, output_dir)
    if args.stage in ("paired-loss", "all"):
        run_paired_loss(cfg, model, dataset, api, samples, output_dir)
    if args.stage == "all":
        run_summary(cfg, output_dir)


if __name__ == "__main__":
    main()
