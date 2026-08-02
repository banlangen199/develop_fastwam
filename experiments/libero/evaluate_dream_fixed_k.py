"""Fixed-K Dream-memory pruning pre-experiment for DreamFastWAM.

This script is deliberately self-contained: it imports the existing model,
checkpoint, dataset, and MoT implementations but does not require any pruning
code to be added to the model.

For every sampled observation it:

1. Creates one action flow-matching training pair (noise and diffusion time).
2. Runs the frozen dense DreamFastWAM and extracts Action-to-Dream attention.
3. Aggregates the configured layers/heads/action tokens into one Dream-token
   saliency ranking.
4. Re-runs the frozen model for each fixed K while masking all unselected
   Dream columns from every Action attention row at every MoT layer.
5. Measures the action flow-matching loss with exactly the same noisy action,
   target, and timestep for every K.

All Dream queries still run through the complete Dream backbone. Dream
decoders and future targets are intentionally not used.

Example:

python experiments/libero/evaluate_dream_fixed_k.py \
  --checkpoint runs/dream_fastwam_libero/RUN/checkpoints/weights/step_008680.pt \
  --config-name dream_fastwam_libero \
  --num-samples 16 \
  --k-values 18,36,54,72,108,144 \
  --attention-layers 20,21,22,23,24,25,26,27,28,29 \
  --output-dir runs/dream_fixed_k_preexperiment

Notes:

* Batch size is intentionally one because the existing debug attention
  extractor currently supports batch size one.
* K=N_D reuses the dense loss, so it does not trigger a duplicate forward.
* This is an offline loss diagnostic, not a success-rate rollout evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import hydra

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fastwam_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fastwam_numba_cache")

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.eval_libero_single import (  # noqa: E402
    _apply_training_model_config,
    _load_model_checkpoint,
    _maybe_load_action_noise_stats,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
)
from fastwam.utils import misc  # noqa: E402


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or not str(raw).strip():
        return None
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


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


def _build_cfg(args: argparse.Namespace) -> DictConfig:
    overrides = [
        f"task={args.config_name}",
        f"ckpt={args.checkpoint}",
        "EVALUATION.num_trials=1",
    ]
    if args.dataset_stats_path:
        overrides.append(f"EVALUATION.dataset_stats_path={args.dataset_stats_path}")
    config_dir = str((PROJECT_ROOT / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="sim_libero.yaml", overrides=overrides)
    if args.device:
        cfg.EVALUATION.device = args.device
    return cfg


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        checkpoint = Path(args.checkpoint)
        step_tag = checkpoint.stem
        run_tag = checkpoint.parents[2].name if len(checkpoint.parents) >= 3 else checkpoint.parent.name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs/dream_fixed_k_preexperiment") / f"{run_tag}_{step_tag}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir.resolve()


def _build_dataset(
    cfg: DictConfig,
    *,
    dataset_stats_path: Path,
    output_dir: Path,
):
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    if data_cfg.get("dream_target") is not None:
        data_cfg.dream_target.enabled = False
    data_cfg.pretrained_norm_stats = str(dataset_stats_path)

    # RobotVideoDataset copies the resolved normalizer statistics into the
    # registered work directory. Keep that diagnostic artifact with this run.
    misc.register_work_dir(output_dir)
    return instantiate(data_cfg, pretrained_norm_stats=str(dataset_stats_path))


def _select_indices(
    *,
    dataset_len: int,
    num_samples: int,
    start_index: int,
    stride: int,
    explicit_indices: list[int] | None,
    sampling: str,
) -> list[int]:
    if dataset_len <= 0:
        raise ValueError("Dataset is empty.")
    if explicit_indices is not None:
        indices = explicit_indices
    elif sampling == "sequential":
        indices = [start_index + offset * stride for offset in range(num_samples)]
    elif sampling == "uniform":
        lo = min(max(start_index, 0), dataset_len - 1)
        available = list(range(lo, dataset_len, stride))
        if not available:
            raise ValueError(
                f"No dataset indices available with start_index={start_index}, stride={stride}, "
                f"dataset_len={dataset_len}."
            )
        if num_samples >= len(available):
            indices = available
        else:
            positions = np.linspace(0, len(available) - 1, num=num_samples, dtype=np.int64)
            indices = [available[int(pos)] for pos in positions]
    else:
        raise ValueError(f"Unknown sampling mode: {sampling}")

    invalid = [index for index in indices if index < 0 or index >= dataset_len]
    if invalid:
        raise ValueError(
            f"Sample indices outside [0, {dataset_len}): {invalid[:20]}"
        )
    if not indices:
        raise ValueError("No sample indices selected.")
    return [int(index) for index in indices]


def _to_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    return int(torch.as_tensor(value).item())


def _sample_metadata(dataset, sample_index: int, num_phase_bins: int) -> dict[str, Any]:
    base_dataset = getattr(dataset, "lerobot_dataset", None)
    if base_dataset is None or not hasattr(base_dataset, "_metadata_from_sample_idx"):
        return {
            "dataset_index": -1,
            "episode_index": -1,
            "frame_index": -1,
            "episode_length": -1,
            "episode_progress": float("nan"),
            "phase_bin": "unknown",
        }

    metadata = base_dataset._metadata_from_sample_idx(sample_index)
    dataset_index = _to_int(metadata.get("dataset_index"))
    episode_index = _to_int(metadata.get("episode_index"))
    frame_index = _to_int(metadata.get("frame_index"))
    episode_length = -1

    try:
        child = base_dataset.multi_dataset._datasets[dataset_index]
        episodes = getattr(child, "episodes", None)
        if episodes is None:
            selected_episode_index = episode_index
        else:
            selected_episode_index = list(episodes).index(episode_index)
        ep_from = int(child.episode_data_index["from"][selected_episode_index].item())
        ep_to = int(child.episode_data_index["to"][selected_episode_index].item())
        episode_length = ep_to - ep_from
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        episode_length = -1

    if episode_length > 1 and frame_index >= 0:
        progress = min(max(frame_index / float(episode_length - 1), 0.0), 1.0)
        bin_index = min(int(progress * num_phase_bins), num_phase_bins - 1)
        phase_bin = f"phase_{bin_index + 1}_of_{num_phase_bins}"
    else:
        progress = float("nan")
        phase_bin = "unknown"

    return {
        "dataset_index": dataset_index,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "episode_length": episode_length,
        "episode_progress": progress,
        "phase_bin": phase_bin,
    }


def _default_k_values(num_dream_tokens: int) -> list[int]:
    ratios = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0)
    return sorted(
        {
            min(max(int(round(ratio * num_dream_tokens)), 1), num_dream_tokens)
            for ratio in ratios
        }
    )


def _validate_k_values(values: Iterable[int], num_dream_tokens: int) -> list[int]:
    k_values = sorted(set(int(value) for value in values))
    if not k_values:
        raise ValueError("At least one K value is required.")
    invalid = [value for value in k_values if value <= 0 or value > num_dream_tokens]
    if invalid:
        raise ValueError(
            f"K values must satisfy 0 < K <= N_D={num_dream_tokens}, got {invalid}."
        )
    if num_dream_tokens not in k_values:
        k_values.append(num_dream_tokens)
        k_values.sort()
        print(f"[fixed-k] Added full-memory K=N_D={num_dream_tokens}.")
    return k_values


def _prepare_flow_matching_pair(model, sample: dict[str, Any], tiled: bool) -> dict[str, Any]:
    inputs = model.build_inputs(sample, tiled=tiled)
    input_latents = inputs["input_latents"]
    batch_size = int(input_latents.shape[0])
    if batch_size != 1:
        raise ValueError(f"This diagnostic requires batch size 1, got {batch_size}.")

    current_frame_latents = inputs["first_frame_latents"]
    if current_frame_latents is None:
        current_frame_latents = input_latents[:, :, 0:1]

    action = inputs["action"]
    noise_action = model._sample_action_noise(
        action,
        use_correlated_noise=model.use_correlated_noise_train,
    )
    timestep_action = model.train_action_scheduler.sample_training_t(
        batch_size=batch_size,
        device=model.device,
        dtype=action.dtype,
    )
    noisy_action = model.train_action_scheduler.add_noise(action, noise_action, timestep_action)
    target_action = model.train_action_scheduler.training_target(action, noise_action, timestep_action)
    timestep_video = torch.zeros(
        (batch_size,),
        device=model.device,
        dtype=current_frame_latents.dtype,
    )
    return {
        "latents_video": current_frame_latents,
        "timestep_video": timestep_video,
        "noisy_action": noisy_action,
        "timestep_action": timestep_action,
        "target_action": target_action,
        "action": action,
        "action_is_pad": inputs["action_is_pad"],
        "context": inputs["context"],
        "context_mask": inputs["context_mask"],
        "fuse_vae_embedding_in_latents": inputs["fuse_vae_embedding_in_latents"],
    }


def _action_loss(
    *,
    model,
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    action_is_pad: torch.Tensor | None,
    timestep_action: torch.Tensor,
) -> torch.Tensor:
    token_loss = F.mse_loss(
        pred_action.float(),
        target_action.float(),
        reduction="none",
    ).mean(dim=2)
    if action_is_pad is None:
        per_sample = token_loss.mean(dim=1)
    else:
        valid = (~action_is_pad).to(device=token_loss.device, dtype=token_loss.dtype)
        per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    weight = model.train_action_scheduler.training_weight(timestep_action).to(
        device=per_sample.device,
        dtype=per_sample.dtype,
    )
    return (per_sample * weight).mean()


def _forward_action(
    *,
    model,
    pair: dict[str, Any],
    selected_dream_indices: torch.Tensor | None,
    capture_attention: bool,
    attention_layers: list[int],
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, int]]:
    video_pre = model.video_expert.pre_dit(
        x=pair["latents_video"],
        timestep=pair["timestep_video"],
        context=pair["context"],
        context_mask=pair["context_mask"],
        action=pair["action"],
        fuse_vae_embedding_in_latents=pair["fuse_vae_embedding_in_latents"],
    )
    dream_pre = model.dream_expert.pre_dit(
        batch_size=1,
        device=video_pre["tokens"].device,
        dtype=video_pre["tokens"].dtype,
        context=pair["context"],
        context_mask=pair["context_mask"],
    )
    action_pre = model.action_expert.pre_dit(
        action_tokens=pair["noisy_action"],
        timestep=pair["timestep_action"],
        context=pair["context"],
        context_mask=pair["context_mask"],
    )

    video_len = int(video_pre["tokens"].shape[1])
    dream_len = int(dream_pre["tokens"].shape[1])
    action_len = int(action_pre["tokens"].shape[1])
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_len,
        dream_seq_len=dream_len,
        action_seq_len=action_len,
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        device=video_pre["tokens"].device,
    ).clone()

    if selected_dream_indices is not None:
        selected_dream_indices = selected_dream_indices.to(
            device=attention_mask.device,
            dtype=torch.long,
        )
        if selected_dream_indices.ndim != 1:
            raise ValueError(
                f"selected_dream_indices must be 1D, got {tuple(selected_dream_indices.shape)}"
            )
        dream_start = video_len
        action_start = video_len + dream_len
        attention_mask[action_start:, dream_start:action_start] = False
        attention_mask[
            action_start:,
            dream_start + selected_dream_indices,
        ] = True

    mot_out = model.mot(
        embeds_all={
            "video": video_pre["tokens"],
            "dream": dream_pre["tokens"],
            "action": action_pre["tokens"],
        },
        attention_mask=attention_mask,
        freqs_all={
            "video": video_pre["freqs"],
            "dream": dream_pre["freqs"],
            "action": action_pre["freqs"],
        },
        context_all={
            "video": {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            "dream": {
                "context": dream_pre["context"],
                "mask": dream_pre["context_mask"],
            },
            "action": {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
        },
        t_mod_all={
            "video": video_pre["t_mod"],
            "dream": dream_pre["t_mod"],
            "action": action_pre["t_mod"],
        },
        return_action_attention=capture_attention,
        attention_layers=attention_layers,
    )
    if capture_attention:
        tokens_out = mot_out["tokens"]
        records = mot_out["action_attention"]
    else:
        tokens_out = mot_out
        records = []
    pred_action = model.action_expert.post_dit(tokens_out["action"], action_pre)
    return pred_action, records, {
        "video": video_len,
        "dream": dream_len,
        "action": action_len,
    }


def _aggregate_teacher_saliency(
    records: list[dict[str, Any]],
    num_dream_tokens: int,
) -> tuple[torch.Tensor, float]:
    vectors = []
    dream_masses = []
    for record in records:
        if "dream" not in record["slices"]:
            continue
        dream_start, dream_end = record["slices"]["dream"]
        probs = record["probs"].float()  # CPU [B,H,Sa,S]
        dream_probs = probs[..., dream_start:dream_end]
        if int(dream_probs.shape[-1]) != num_dream_tokens:
            raise ValueError(
                "Captured Dream attention length mismatch: "
                f"expected {num_dream_tokens}, got {dream_probs.shape[-1]}."
            )
        vectors.append(dream_probs.mean(dim=(0, 1, 2)))
        dream_masses.append(float(dream_probs.sum(dim=-1).mean().item()))
    if not vectors:
        raise ValueError("No Action-to-Dream attention records were captured.")

    # Aggregate raw probabilities first, then normalize once over Dream. Layers
    # that assign negligible total mass to Dream naturally contribute less.
    saliency = torch.stack(vectors, dim=0).mean(dim=0)
    saliency_sum = saliency.sum()
    if not torch.isfinite(saliency_sum) or float(saliency_sum.item()) <= 0.0:
        raise ValueError("Teacher Dream saliency is empty or non-finite.")
    saliency = saliency / saliency_sum
    return saliency, float(np.mean(dream_masses))


def _dream_token_metadata(model, num_dream_tokens: int) -> list[dict[str, Any]]:
    modalities = list(model.dream_expert.modalities)
    offsets = list(model.dream_expert.future_offsets)
    slices = model.dream_expert.modality_slices()
    result: list[dict[str, Any]] = [dict() for _ in range(num_dream_tokens)]
    for modality_id, modality in enumerate(modalities):
        slc = slices[modality]
        count = int(slc.stop - slc.start)
        if len(offsets) <= 0 or count % len(offsets) != 0:
            raise ValueError(
                f"Cannot derive metadata for modality={modality}: "
                f"slice length={count}, num_offsets={len(offsets)}."
            )
        slots_per_horizon = count // len(offsets)
        for horizon_id, offset in enumerate(offsets):
            for slot_id in range(slots_per_horizon):
                token_index = int(slc.start) + horizon_id * slots_per_horizon + slot_id
                result[token_index] = {
                    "token_index": token_index,
                    "modality": str(modality),
                    "modality_id": modality_id,
                    "horizon_id": horizon_id,
                    "future_offset": int(offset),
                    "slot_id": slot_id,
                }
    if any(not item for item in result):
        raise ValueError("Dream token metadata did not cover every token identity.")
    return result


def _retention_summary(
    selected_indices: torch.Tensor,
    token_metadata: list[dict[str, Any]],
) -> dict[str, float]:
    selected = {int(index) for index in selected_indices.tolist()}
    groups: dict[str, list[int]] = {}
    for item in token_metadata:
        groups.setdefault(f"modality_{item['modality']}", []).append(int(item["token_index"]))
        groups.setdefault(f"horizon_{item['future_offset']}", []).append(int(item["token_index"]))
    return {
        key: sum(index in selected for index in indices) / float(len(indices))
        for key, indices in groups.items()
    }


def _mean_std_sem(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "sem": float("nan")}
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": std,
        "sem": std / math.sqrt(float(array.size)),
    }


def _summarize(rows: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "by_phase": {}}
    for k in k_values:
        losses = [float(row[f"loss_k{k}"]) for row in rows]
        deltas = [float(row[f"delta_k{k}"]) for row in rows]
        relatives = [float(row[f"relative_delta_k{k}"]) for row in rows]
        summary["overall"][str(k)] = {
            "loss": _mean_std_sem(losses),
            "delta_vs_full": _mean_std_sem(deltas),
            "relative_delta_vs_full": _mean_std_sem(relatives),
        }

    phase_names = sorted({str(row["phase_bin"]) for row in rows})
    for phase_name in phase_names:
        phase_rows = [row for row in rows if str(row["phase_bin"]) == phase_name]
        summary["by_phase"][phase_name] = {}
        for k in k_values:
            summary["by_phase"][phase_name][str(k)] = {
                "loss": _mean_std_sem([float(row[f"loss_k{k}"]) for row in phase_rows]),
                "delta_vs_full": _mean_std_sem([float(row[f"delta_k{k}"]) for row in phase_rows]),
            }
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_plots(output_dir: Path, summary: dict[str, Any], k_values: list[int]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fixed-k] matplotlib is unavailable; skipping plots.")
        return

    means = [summary["overall"][str(k)]["loss"]["mean"] for k in k_values]
    sems = [summary["overall"][str(k)]["loss"]["sem"] for k in k_values]
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(k_values, means, yerr=sems, marker="o", capsize=3)
    plt.xlabel("Retained Dream tokens (K)")
    plt.ylabel("Action flow-matching loss")
    plt.title("Fixed-K Dream pruning")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "action_loss_vs_k.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for phase_name, phase_values in summary["by_phase"].items():
        if phase_name == "unknown":
            continue
        phase_means = [phase_values[str(k)]["loss"]["mean"] for k in k_values]
        plt.plot(k_values, phase_means, marker="o", label=phase_name)
    plt.xlabel("Retained Dream tokens (K)")
    plt.ylabel("Action flow-matching loss")
    plt.title("Fixed-K loss by normalized episode progress")
    plt.grid(alpha=0.25)
    if len(summary["by_phase"]) > 1 or "unknown" not in summary["by_phase"]:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "action_loss_vs_k_by_phase.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure frozen DreamFastWAM action loss under attention-ranked fixed-K Dream pruning."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config-name",
        required=True,
        help="Hydra task config name, e.g. dream_fastwam_libero or dream_fastwam_libero_spatial.",
    )
    parser.add_argument("--dataset-stats-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sample-indices", default=None, help="Explicit comma-separated dataset indices.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--sampling", choices=["uniform", "sequential"], default="uniform")
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--k-values", default=None, help="Comma-separated absolute K values; full K is added.")
    parser.add_argument(
        "--attention-layers",
        default=None,
        help="Comma-separated MoT layers. Default: all layers.",
    )
    parser.add_argument("--num-phase-bins", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiled", action="store_true")
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.noise_repeats <= 0:
        raise ValueError("--noise-repeats must be positive.")
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be positive.")
    if args.num_phase_bins <= 0:
        raise ValueError("--num-phase-bins must be positive.")

    _set_seed(args.seed)
    output_dir = _resolve_output_dir(args)
    cfg = _build_cfg(args)
    training_config_path = _apply_training_model_config(cfg)
    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    device = str(args.device or cfg.EVALUATION.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = _mixed_precision_to_model_dtype(str(cfg.get("mixed_precision", "bf16")))

    # Cached dataset contexts are used, so the text encoder is unnecessary for
    # this diagnostic and would only consume memory.
    cfg.model.load_text_encoder = False
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    _load_model_checkpoint(model, args.checkpoint)
    _maybe_load_action_noise_stats(model, cfg, dataset_stats_path)
    model = model.to(device).eval()
    model.requires_grad_(False)

    if not hasattr(model, "dream_expert"):
        raise TypeError("This script requires DreamFastWAM with a dream_expert.")
    if list(getattr(model.mot, "expert_order", [])) != ["video", "dream", "action"]:
        raise ValueError(
            "Expected MoT expert order ['video', 'dream', 'action'], got "
            f"{getattr(model.mot, 'expert_order', None)}."
        )

    dataset = _build_dataset(
        cfg,
        dataset_stats_path=dataset_stats_path,
        output_dir=output_dir,
    )
    indices = _select_indices(
        dataset_len=len(dataset),
        num_samples=args.num_samples,
        start_index=args.start_index,
        stride=args.sample_stride,
        explicit_indices=_parse_int_list(args.sample_indices),
        sampling=args.sampling,
    )
    total_layers = int(model.mot.num_layers)
    attention_layers = _parse_int_list(args.attention_layers) or list(range(total_layers))
    invalid_layers = [layer for layer in attention_layers if layer < 0 or layer >= total_layers]
    if invalid_layers:
        raise ValueError(f"Attention layers outside [0, {total_layers}): {invalid_layers}")

    # Query count is input-independent, so one lightweight pre_dit call is
    # unnecessary: derive N_D from modality slices exactly as the model does.
    modality_slices = model.dream_expert.modality_slices()
    num_dream_tokens = max(int(slc.stop) for slc in modality_slices.values())
    requested_k_values = _parse_int_list(args.k_values)
    k_values = _validate_k_values(
        requested_k_values or _default_k_values(num_dream_tokens),
        num_dream_tokens,
    )
    token_metadata = _dream_token_metadata(model, num_dream_tokens)

    run_info = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config_name": args.config_name,
        "training_config_path": None if training_config_path is None else str(training_config_path),
        "dataset_stats_path": str(dataset_stats_path),
        "output_dir": str(output_dir),
        "device": device,
        "model_dtype": str(model_dtype),
        "sample_indices": indices,
        "noise_repeats": args.noise_repeats,
        "attention_layers": attention_layers,
        "k_values": k_values,
        "num_dream_tokens": num_dream_tokens,
        "num_phase_bins": args.num_phase_bins,
        "seed": args.seed,
        "dream_decoder_executed": False,
        "future_targets_loaded": False,
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(_jsonable(run_info), indent=2),
        encoding="utf-8",
    )
    (output_dir / "dream_token_metadata.json").write_text(
        json.dumps(token_metadata, indent=2),
        encoding="utf-8",
    )
    OmegaConf.save(config=cfg, f=str(output_dir / "resolved_config.yaml"))

    print(f"[fixed-k] output_dir={output_dir}")
    print(f"[fixed-k] samples={len(indices)} noise_repeats={args.noise_repeats}")
    print(f"[fixed-k] N_D={num_dream_tokens} K={k_values}")
    print(f"[fixed-k] attention_layers={attention_layers}")
    print("[fixed-k] Dream decoders/future targets are disabled.")

    rows: list[dict[str, Any]] = []
    saliencies: list[np.ndarray] = []
    rankings: list[np.ndarray] = []
    selected_by_k: dict[int, list[np.ndarray]] = {k: [] for k in k_values}

    with torch.inference_mode():
        for sample_position, sample_index in enumerate(indices):
            metadata = _sample_metadata(dataset, sample_index, args.num_phase_bins)
            sample = default_collate([dataset[sample_index]])
            for repeat_index in range(args.noise_repeats):
                # Make each observation/repeat reproducible while preserving the
                # same noise/timestep for all K comparisons inside this pair.
                pair_seed = args.seed + sample_position * 1009 + repeat_index
                _set_seed(pair_seed)
                pair = _prepare_flow_matching_pair(model, sample, tiled=args.tiled)

                dense_pred, attention_records, seq_lens = _forward_action(
                    model=model,
                    pair=pair,
                    selected_dream_indices=None,
                    capture_attention=True,
                    attention_layers=attention_layers,
                )
                if seq_lens["dream"] != num_dream_tokens:
                    raise ValueError(
                        f"Runtime Dream length={seq_lens['dream']} differs from metadata N_D={num_dream_tokens}."
                    )
                dense_loss = _action_loss(
                    model=model,
                    pred_action=dense_pred,
                    target_action=pair["target_action"],
                    action_is_pad=pair["action_is_pad"],
                    timestep_action=pair["timestep_action"],
                )
                saliency, teacher_dream_mass = _aggregate_teacher_saliency(
                    attention_records,
                    num_dream_tokens,
                )
                ranking = torch.argsort(saliency, descending=True)

                row: dict[str, Any] = {
                    "sample_position": sample_position,
                    "sample_index": sample_index,
                    "repeat_index": repeat_index,
                    **metadata,
                    "pair_seed": pair_seed,
                    "action_timestep": float(pair["timestep_action"][0].float().item()),
                    "teacher_dream_attention_mass": teacher_dream_mass,
                    "dense_loss": float(dense_loss.item()),
                }
                saliencies.append(saliency.numpy())
                rankings.append(ranking.numpy())

                loss_by_k: dict[int, float] = {}
                for k in k_values:
                    selected = ranking[:k]
                    selected_by_k[k].append(selected.numpy())
                    if k == num_dream_tokens:
                        loss_k = dense_loss
                    else:
                        pred_k, _, _ = _forward_action(
                            model=model,
                            pair=pair,
                            selected_dream_indices=selected,
                            capture_attention=False,
                            attention_layers=attention_layers,
                        )
                        loss_k = _action_loss(
                            model=model,
                            pred_action=pred_k,
                            target_action=pair["target_action"],
                            action_is_pad=pair["action_is_pad"],
                            timestep_action=pair["timestep_action"],
                        )
                    loss_value = float(loss_k.item())
                    loss_by_k[k] = loss_value
                    delta = loss_value - float(dense_loss.item())
                    row[f"loss_k{k}"] = loss_value
                    row[f"delta_k{k}"] = delta
                    row[f"relative_delta_k{k}"] = delta / max(abs(float(dense_loss.item())), 1e-8)
                    for group, retention in _retention_summary(selected, token_metadata).items():
                        row[f"retention_{group}_k{k}"] = retention

                rows.append(row)
                compact_losses = " ".join(f"K{k}={loss_by_k[k]:.6f}" for k in k_values)
                print(
                    f"[fixed-k] sample={sample_position + 1}/{len(indices)} "
                    f"idx={sample_index} repeat={repeat_index + 1}/{args.noise_repeats} "
                    f"phase={metadata['phase_bin']} dense={float(dense_loss.item()):.6f} "
                    f"{compact_losses}"
                )

            # Persist progress after every observation so a long diagnostic can
            # be inspected even if it is interrupted.
            _write_csv(output_dir / "per_sample_results.csv", rows)

    summary = _summarize(rows, k_values)
    summary["run_info"] = run_info
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2),
        encoding="utf-8",
    )
    npz_payload: dict[str, np.ndarray] = {
        "teacher_saliency": np.stack(saliencies),
        "teacher_ranking": np.stack(rankings),
    }
    for k, values in selected_by_k.items():
        npz_payload[f"selected_indices_k{k}"] = np.stack(values)
    np.savez_compressed(output_dir / "raw_token_selections.npz", **npz_payload)
    _save_plots(output_dir, summary, k_values)

    print(f"[fixed-k] Saved {output_dir / 'per_sample_results.csv'}")
    print(f"[fixed-k] Saved {output_dir / 'summary.json'}")
    print(f"[fixed-k] Saved {output_dir / 'raw_token_selections.npz'}")
    print(f"[fixed-k] Saved plots under {output_dir}")


if __name__ == "__main__":
    main()
