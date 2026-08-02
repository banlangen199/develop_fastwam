"""Three-group layer-wise Dream pruning pre-experiment.

This script complements ``evaluate_dream_fixed_k.py`` without changing it or
any model implementation. The Action layers are split into three contiguous
groups:

* bottom: early Action layers
* middle: middle Action layers
* high: late Action layers

Each group aggregates its own Dense Action-to-Dream attention and obtains its
own Dream-token ranking. A configuration is a triplet
``bottom_K:middle_K:high_K``. Every layer in one group uses the same selected
identities, while different groups may use different K values and identities.

The default 30-layer split is bottom=0..9, middle=10..19, high=20..29.
The default candidate values ``72,108,144`` generate only 11 controlled
configurations instead of the full 3^3 Cartesian product:

* full memory
* same K in all groups
* vary one group while the other two remain full
* two directional capacity schedules

Custom triplets can be supplied explicitly.

Example:

python experiments/libero/evaluate_dream_layer_group_k.py \
  --checkpoint runs/dream_fastwam_libero/RUN/checkpoints/weights/step_008680.pt \
  --config-name dream_fastwam_libero \
  --num-samples 8 \
  --noise-repeats 1 \
  --k-values 72,108,144 \
  --output-dir runs/dream_layer_group_k/smoke

Custom configurations:

python experiments/libero/evaluate_dream_layer_group_k.py \
  ... \
  --triplets 72:72:72,108:108:108,72:108:144,144:108:72

All Dream queries still execute the complete Dream backbone. Dream decoders
and future targets are disabled. Pruning is implemented as exact Action-row
attention masking for this offline loss diagnostic; it does not physically
shorten K/V tensors or measure inference speed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fastwam_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fastwam_numba_cache")

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
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
from experiments.libero.evaluate_dream_fixed_k import (  # noqa: E402
    _action_loss,
    _aggregate_teacher_saliency,
    _build_cfg,
    _build_dataset,
    _dream_token_metadata,
    _forward_action,
    _parse_int_list,
    _prepare_flow_matching_pair,
    _retention_summary,
    _sample_metadata,
    _select_indices,
    _set_seed,
    _validate_k_values,
)


GROUP_NAMES = ("bottom", "middle", "high")


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        checkpoint = Path(args.checkpoint)
        step_tag = checkpoint.stem
        run_tag = checkpoint.parents[2].name if len(checkpoint.parents) >= 3 else checkpoint.parent.name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs/dream_layer_group_k") / f"{run_tag}_{step_tag}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir.resolve()


def _build_layer_groups(
    num_layers: int,
    boundaries: list[int] | None,
) -> dict[str, list[int]]:
    if num_layers < 3:
        raise ValueError(f"At least three MoT layers are required, got {num_layers}.")
    if boundaries is None:
        boundaries = [num_layers // 3, (2 * num_layers) // 3]
    if len(boundaries) != 2:
        raise ValueError(
            f"--layer-boundaries requires exactly two integers, got {boundaries}."
        )
    first, second = (int(boundaries[0]), int(boundaries[1]))
    if not (0 < first < second < num_layers):
        raise ValueError(
            "Layer boundaries must satisfy "
            f"0 < first < second < num_layers={num_layers}, got {boundaries}."
        )
    groups = {
        "bottom": list(range(0, first)),
        "middle": list(range(first, second)),
        "high": list(range(second, num_layers)),
    }
    covered = [layer for name in GROUP_NAMES for layer in groups[name]]
    if covered != list(range(num_layers)):
        raise AssertionError(f"Layer groups do not exactly cover all layers: {groups}")
    return groups


def _parse_triplets(raw: str | None, num_dream_tokens: int) -> list[tuple[int, int, int]] | None:
    if raw is None or not str(raw).strip():
        return None
    triplets = []
    for item in str(raw).split(","):
        values = [int(value.strip()) for value in item.split(":") if value.strip()]
        if len(values) != 3:
            raise ValueError(
                "Each --triplets item must be bottom:middle:high, "
                f"got {item!r}."
            )
        invalid = [value for value in values if value <= 0 or value > num_dream_tokens]
        if invalid:
            raise ValueError(
                f"Triplet {values} contains K outside (0, N_D={num_dream_tokens}]."
            )
        triplets.append(tuple(values))
    return _deduplicate_triplets(
        [(num_dream_tokens, num_dream_tokens, num_dream_tokens), *triplets]
    )


def _deduplicate_triplets(
    triplets: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    result = []
    seen = set()
    for triplet in triplets:
        triplet = tuple(int(value) for value in triplet)
        if triplet in seen:
            continue
        seen.add(triplet)
        result.append(triplet)
    return result


def _default_triplets(
    k_values: list[int],
    num_dream_tokens: int,
) -> list[tuple[int, int, int]]:
    full = int(num_dream_tokens)
    reduced = [value for value in k_values if value < full]
    triplets: list[tuple[int, int, int]] = [(full, full, full)]

    # Same capacity but independent identities in all three groups.
    triplets.extend((value, value, value) for value in reduced)

    # Change one group at a time to measure where pruning is most harmful.
    for value in reduced:
        triplets.extend(
            [
                (value, full, full),
                (full, value, full),
                (full, full, value),
            ]
        )

    # Compare increasing versus decreasing capacity with depth.
    if len(reduced) >= 2:
        low = reduced[0]
        high = reduced[-1]
        triplets.extend(
            [
                (low, high, full),
                (full, high, low),
            ]
        )
    return _deduplicate_triplets(triplets)


def _triplet_key(triplet: tuple[int, int, int]) -> str:
    return f"b{triplet[0]}_m{triplet[1]}_h{triplet[2]}"


def _records_for_layers(
    records: list[dict[str, Any]],
    layers: list[int],
) -> list[dict[str, Any]]:
    selected = [record for record in records if int(record["layer"]) in set(layers)]
    found = {int(record["layer"]) for record in selected}
    missing = sorted(set(layers) - found)
    if missing:
        raise ValueError(f"Missing attention records for layers: {missing}")
    return selected


@contextmanager
def _patch_layerwise_action_dream_masks(
    *,
    mot,
    layer_to_selected: dict[int, torch.Tensor],
    video_seq_len: int,
    dream_seq_len: int,
    expected_num_layers: int,
) -> Iterator[dict[str, int]]:
    """Temporarily choose a different Action-to-Dream mask at each MoT layer."""

    original_bound_method = mot._mixed_attention
    had_instance_override = "_mixed_attention" in mot.__dict__
    previous_instance_value = mot.__dict__.get("_mixed_attention")
    state = {"calls": 0}
    dream_start = int(video_seq_len)
    action_start = int(video_seq_len + dream_seq_len)

    def layerwise_mixed_attention(
        _mot,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        layer_index = state["calls"]
        state["calls"] += 1
        if layer_index not in layer_to_selected:
            raise RuntimeError(
                f"No selected Dream identities configured for MoT layer {layer_index}."
            )
        selected = layer_to_selected[layer_index].to(
            device=attention_mask.device,
            dtype=torch.long,
        )
        layer_mask = attention_mask.clone()
        layer_mask[action_start:, dream_start:action_start] = False
        layer_mask[action_start:, dream_start + selected] = True
        return original_bound_method(
            q_cat=q_cat,
            k_cat=k_cat,
            v_cat=v_cat,
            attention_mask=layer_mask,
        )

    mot._mixed_attention = types.MethodType(layerwise_mixed_attention, mot)
    try:
        yield state
    finally:
        if had_instance_override:
            mot._mixed_attention = previous_instance_value
        else:
            delattr(mot, "_mixed_attention")
        if state["calls"] != expected_num_layers:
            raise RuntimeError(
                "Layer-wise mask hook call count mismatch: "
                f"expected {expected_num_layers}, got {state['calls']}."
            )


def _forward_action_layer_group_pruned(
    *,
    model,
    pair: dict[str, Any],
    rankings: dict[str, torch.Tensor],
    layer_groups: dict[str, list[int]],
    triplet: tuple[int, int, int],
    seq_lens: dict[str, int],
) -> torch.Tensor:
    group_selected = {
        name: rankings[name][: int(k)]
        for name, k in zip(GROUP_NAMES, triplet)
    }
    layer_to_selected = {
        layer: group_selected[name]
        for name in GROUP_NAMES
        for layer in layer_groups[name]
    }
    with _patch_layerwise_action_dream_masks(
        mot=model.mot,
        layer_to_selected=layer_to_selected,
        video_seq_len=seq_lens["video"],
        dream_seq_len=seq_lens["dream"],
        expected_num_layers=int(model.mot.num_layers),
    ):
        pred_action, _, routed_seq_lens = _forward_action(
            model=model,
            pair=pair,
            selected_dream_indices=None,
            capture_attention=False,
            attention_layers=[],
        )
    if routed_seq_lens != seq_lens:
        raise ValueError(
            f"Sequence lengths changed between dense and layer-wise forwards: "
            f"dense={seq_lens}, routed={routed_seq_lens}."
        )
    return pred_action


def _group_overlap(
    rankings: dict[str, torch.Tensor],
    triplet: tuple[int, int, int],
) -> dict[str, float]:
    selected = {
        name: set(int(index) for index in rankings[name][:k].tolist())
        for name, k in zip(GROUP_NAMES, triplet)
    }
    result = {}
    for left, right in (("bottom", "middle"), ("middle", "high"), ("bottom", "high")):
        union = selected[left] | selected[right]
        result[f"jaccard_{left}_{right}"] = (
            len(selected[left] & selected[right]) / float(len(union))
            if union
            else 1.0
        )
    return result


def _mean_std_sem(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
        }
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": std,
        "sem": std / math.sqrt(float(array.size)),
    }


def _summarize_clustered(
    rows: list[dict[str, Any]],
    triplets: list[tuple[int, int, int]],
) -> dict[str, Any]:
    """Treat each observation, rather than each noise repeat, as one unit."""

    sample_positions = sorted({int(row["sample_position"]) for row in rows})
    result: dict[str, Any] = {
        "num_observations": len(sample_positions),
        "num_rows_with_repeats": len(rows),
        "configurations": {},
        "by_phase": {},
    }
    for triplet in triplets:
        key = _triplet_key(triplet)
        observation_losses = []
        observation_deltas = []
        observation_relative_deltas = []
        for sample_position in sample_positions:
            sample_rows = [
                row for row in rows
                if int(row["sample_position"]) == sample_position
            ]
            observation_losses.append(
                float(np.mean([float(row[f"loss_{key}"]) for row in sample_rows]))
            )
            observation_deltas.append(
                float(np.mean([float(row[f"delta_{key}"]) for row in sample_rows]))
            )
            observation_relative_deltas.append(
                float(np.mean([float(row[f"relative_delta_{key}"]) for row in sample_rows]))
            )
        result["configurations"][key] = {
            "triplet": {
                "bottom": triplet[0],
                "middle": triplet[1],
                "high": triplet[2],
            },
            "loss": _mean_std_sem(observation_losses),
            "delta_vs_full": _mean_std_sem(observation_deltas),
            "relative_delta_vs_full": _mean_std_sem(observation_relative_deltas),
            "num_observations_better_than_full": int(
                np.sum(np.asarray(observation_deltas) < 0.0)
            ),
        }

    phase_names = sorted({str(row["phase_bin"]) for row in rows})
    for phase_name in phase_names:
        phase_rows = [row for row in rows if str(row["phase_bin"]) == phase_name]
        phase_positions = sorted({int(row["sample_position"]) for row in phase_rows})
        result["by_phase"][phase_name] = {
            "num_observations": len(phase_positions),
            "configurations": {},
        }
        for triplet in triplets:
            key = _triplet_key(triplet)
            phase_observation_losses = []
            for sample_position in phase_positions:
                sample_rows = [
                    row for row in phase_rows
                    if int(row["sample_position"]) == sample_position
                ]
                phase_observation_losses.append(
                    float(np.mean([float(row[f"loss_{key}"]) for row in sample_rows]))
                )
            result["by_phase"][phase_name]["configurations"][key] = {
                "loss": _mean_std_sem(phase_observation_losses),
            }
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(
    output_dir: Path,
    summary: dict[str, Any],
    triplets: list[tuple[int, int, int]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[layer-group-k] matplotlib is unavailable; skipping plot.")
        return

    keys = [_triplet_key(triplet) for triplet in triplets]
    means = [
        summary["configurations"][key]["relative_delta_vs_full"]["mean"]
        for key in keys
    ]
    sems = [
        summary["configurations"][key]["relative_delta_vs_full"]["sem"]
        for key in keys
    ]
    figure_width = max(9.0, len(keys) * 0.85)
    plt.figure(figsize=(figure_width, 5.0))
    positions = np.arange(len(keys))
    plt.bar(positions, means, yerr=sems, capsize=3)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(positions, keys, rotation=45, ha="right")
    plt.ylabel("Relative action-loss change vs full")
    plt.title("Layer-group Dream pruning (clustered by observation)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "layer_group_relative_loss.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate bottom/middle/high Action layer groups with independent Dream Top-K."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--dataset-stats-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-indices", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--sampling", choices=["uniform", "sequential"], default="uniform")
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument(
        "--k-values",
        default="72,108,144",
        help="Candidate K values used to generate controlled default triplets.",
    )
    parser.add_argument(
        "--triplets",
        default=None,
        help="Optional bottom:middle:high triplets separated by commas.",
    )
    parser.add_argument(
        "--layer-boundaries",
        default=None,
        help="Two comma-separated split points. Default for 30 layers: 10,20.",
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
    device = str(
        args.device
        or cfg.EVALUATION.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_dtype = _mixed_precision_to_model_dtype(str(cfg.get("mixed_precision", "bf16")))
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
    num_layers = int(model.mot.num_layers)
    layer_groups = _build_layer_groups(
        num_layers,
        _parse_int_list(args.layer_boundaries),
    )
    modality_slices = model.dream_expert.modality_slices()
    num_dream_tokens = max(int(slc.stop) for slc in modality_slices.values())
    k_values = _validate_k_values(
        _parse_int_list(args.k_values) or [72, 108, num_dream_tokens],
        num_dream_tokens,
    )
    triplets = _parse_triplets(args.triplets, num_dream_tokens)
    if triplets is None:
        triplets = _default_triplets(k_values, num_dream_tokens)
    full_triplet = (num_dream_tokens, num_dream_tokens, num_dream_tokens)
    if full_triplet not in triplets:
        triplets.insert(0, full_triplet)
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
        "num_layers": num_layers,
        "layer_groups": layer_groups,
        "k_values": k_values,
        "triplets": [
            {
                "key": _triplet_key(triplet),
                "bottom": triplet[0],
                "middle": triplet[1],
                "high": triplet[2],
            }
            for triplet in triplets
        ],
        "num_dream_tokens": num_dream_tokens,
        "num_phase_bins": args.num_phase_bins,
        "seed": args.seed,
        "dream_decoder_executed": False,
        "future_targets_loaded": False,
        "physical_kv_gather": False,
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2),
        encoding="utf-8",
    )
    (output_dir / "dream_token_metadata.json").write_text(
        json.dumps(token_metadata, indent=2),
        encoding="utf-8",
    )
    OmegaConf.save(config=cfg, f=str(output_dir / "resolved_config.yaml"))

    print(f"[layer-group-k] output_dir={output_dir}")
    print(f"[layer-group-k] layer_groups={layer_groups}")
    print(f"[layer-group-k] N_D={num_dream_tokens}")
    print(
        "[layer-group-k] configurations="
        + ",".join(_triplet_key(triplet) for triplet in triplets)
    )
    print("[layer-group-k] Dream decoders/future targets are disabled.")

    rows: list[dict[str, Any]] = []
    saliency_by_group: dict[str, list[np.ndarray]] = {
        name: [] for name in GROUP_NAMES
    }
    ranking_by_group: dict[str, list[np.ndarray]] = {
        name: [] for name in GROUP_NAMES
    }

    with torch.inference_mode():
        for sample_position, sample_index in enumerate(indices):
            metadata = _sample_metadata(dataset, sample_index, args.num_phase_bins)
            sample = default_collate([dataset[sample_index]])
            for repeat_index in range(args.noise_repeats):
                pair_seed = args.seed + sample_position * 1009 + repeat_index
                _set_seed(pair_seed)
                pair = _prepare_flow_matching_pair(model, sample, tiled=args.tiled)
                dense_pred, attention_records, seq_lens = _forward_action(
                    model=model,
                    pair=pair,
                    selected_dream_indices=None,
                    capture_attention=True,
                    attention_layers=list(range(num_layers)),
                )
                if seq_lens["dream"] != num_dream_tokens:
                    raise ValueError(
                        f"Runtime Dream length={seq_lens['dream']} differs from N_D={num_dream_tokens}."
                    )
                dense_loss = _action_loss(
                    model=model,
                    pred_action=dense_pred,
                    target_action=pair["target_action"],
                    action_is_pad=pair["action_is_pad"],
                    timestep_action=pair["timestep_action"],
                )

                group_saliency: dict[str, torch.Tensor] = {}
                group_rankings: dict[str, torch.Tensor] = {}
                group_dream_masses: dict[str, float] = {}
                for name in GROUP_NAMES:
                    saliency, dream_mass = _aggregate_teacher_saliency(
                        _records_for_layers(attention_records, layer_groups[name]),
                        num_dream_tokens,
                    )
                    ranking = torch.argsort(saliency, descending=True)
                    group_saliency[name] = saliency
                    group_rankings[name] = ranking
                    group_dream_masses[name] = dream_mass
                    saliency_by_group[name].append(saliency.numpy())
                    ranking_by_group[name].append(ranking.numpy())

                row: dict[str, Any] = {
                    "sample_position": sample_position,
                    "sample_index": sample_index,
                    "repeat_index": repeat_index,
                    **metadata,
                    "pair_seed": pair_seed,
                    "action_timestep": float(pair["timestep_action"][0].float().item()),
                    "dense_loss": float(dense_loss.item()),
                }
                for name in GROUP_NAMES:
                    row[f"teacher_dream_attention_mass_{name}"] = group_dream_masses[name]

                loss_items = []
                for triplet in triplets:
                    key = _triplet_key(triplet)
                    if triplet == full_triplet:
                        loss = dense_loss
                    else:
                        pred_action = _forward_action_layer_group_pruned(
                            model=model,
                            pair=pair,
                            rankings=group_rankings,
                            layer_groups=layer_groups,
                            triplet=triplet,
                            seq_lens=seq_lens,
                        )
                        loss = _action_loss(
                            model=model,
                            pred_action=pred_action,
                            target_action=pair["target_action"],
                            action_is_pad=pair["action_is_pad"],
                            timestep_action=pair["timestep_action"],
                        )
                    loss_value = float(loss.item())
                    delta = loss_value - float(dense_loss.item())
                    row[f"loss_{key}"] = loss_value
                    row[f"delta_{key}"] = delta
                    row[f"relative_delta_{key}"] = (
                        delta / max(abs(float(dense_loss.item())), 1e-8)
                    )
                    for overlap_name, overlap in _group_overlap(
                        group_rankings,
                        triplet,
                    ).items():
                        row[f"{overlap_name}_{key}"] = overlap
                    for group_name, group_k in zip(GROUP_NAMES, triplet):
                        selected = group_rankings[group_name][:group_k]
                        for retention_name, retention in _retention_summary(
                            selected,
                            token_metadata,
                        ).items():
                            row[
                                f"retention_{group_name}_{retention_name}_{key}"
                            ] = retention
                    loss_items.append(f"{key}={loss_value:.6f}")

                rows.append(row)
                print(
                    f"[layer-group-k] sample={sample_position + 1}/{len(indices)} "
                    f"idx={sample_index} repeat={repeat_index + 1}/{args.noise_repeats} "
                    f"dense={float(dense_loss.item()):.6f} "
                    + " ".join(loss_items)
                )
            _write_csv(output_dir / "per_sample_results.csv", rows)

    summary = _summarize_clustered(rows, triplets)
    summary["run_info"] = run_info
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    npz_payload = {}
    for name in GROUP_NAMES:
        npz_payload[f"teacher_saliency_{name}"] = np.stack(saliency_by_group[name])
        npz_payload[f"teacher_ranking_{name}"] = np.stack(ranking_by_group[name])
    np.savez_compressed(
        output_dir / "raw_layer_group_rankings.npz",
        **npz_payload,
    )
    _save_plot(output_dir, summary, triplets)

    print(f"[layer-group-k] Saved {output_dir / 'per_sample_results.csv'}")
    print(f"[layer-group-k] Saved {output_dir / 'summary.json'}")
    print(f"[layer-group-k] Saved {output_dir / 'raw_layer_group_rankings.npz'}")
    print(f"[layer-group-k] Saved {output_dir / 'layer_group_relative_loss.png'}")


if __name__ == "__main__":
    main()
