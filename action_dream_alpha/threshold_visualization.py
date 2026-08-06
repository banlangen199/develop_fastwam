from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fastwam_threshold_eval_matplotlib")


def reconstruct_threshold_attention(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Reconstruct the exact post-threshold distribution from a debug record.

    The model record contains the original full mixed-attention probability and
    the Dream keep mask. Zeroing rejected Dream columns and normalizing over all
    remaining mixed keys is mathematically identical to the model's second
    masked-logit softmax.
    """
    probs = torch.as_tensor(record["probs"]).detach().cpu().float()
    if probs.ndim != 4 or probs.shape[0] != 1:
        raise ValueError(f"Expected probabilities [1,H,A,S], got {tuple(probs.shape)}")
    probs = probs[0]
    slices = record["slices"]
    dream_start, dream_end = (int(x) for x in slices["dream"])
    threshold = record.get("threshold")
    if not isinstance(threshold, dict):
        raise ValueError("Threshold debug record is missing `threshold` details.")
    keep = torch.as_tensor(threshold["keep_mask"]).detach().cpu().bool()
    score = torch.as_tensor(threshold["score"]).detach().cpu().float()
    if keep.ndim == 2:
        if keep.shape[0] != 1:
            raise ValueError("Visualization currently expects batch size one.")
        keep = keep[0]
    if score.ndim == 2:
        if score.shape[0] != 1:
            raise ValueError("Visualization currently expects batch size one.")
        score = score[0]
    num_dream = dream_end - dream_start
    if keep.shape != (num_dream,) or score.shape != (num_dream,):
        raise ValueError(
            f"Dream details must have shape {(num_dream,)}, got keep={tuple(keep.shape)}, "
            f"score={tuple(score.shape)}"
        )
    dense_sum_error = (probs.sum(dim=-1) - 1.0).abs().amax()
    key_keep = torch.ones(probs.shape[-1], dtype=torch.bool)
    key_keep[dream_start:dream_end] = keep
    pruned = probs * key_keep[None, None, :]
    denominator = pruned.sum(dim=-1, keepdim=True)
    if bool((denominator <= 0).any()):
        raise FloatingPointError("Threshold mask removed every visible mixed-attention key.")
    pruned = pruned / denominator
    pruned_sum_error = (pruned.sum(dim=-1) - 1.0).abs().amax()

    dense_dream = probs[..., dream_start:dream_end]
    pruned_dream = pruned[..., dream_start:dream_end]
    return {
        "dense_probs": probs,
        "pruned_probs": pruned,
        "dense_dream_headwise": dense_dream,
        "pruned_dream_headwise": pruned_dream,
        "dense_dream_by_action": dense_dream.mean(dim=0),
        "pruned_dream_by_action": pruned_dream.mean(dim=0),
        "dense_dream_by_token": dense_dream.mean(dim=(0, 1)),
        "pruned_dream_by_token": pruned_dream.mean(dim=(0, 1)),
        "dense_all_by_token": probs.mean(dim=(0, 1)),
        "pruned_all_by_token": pruned.mean(dim=(0, 1)),
        "score": score,
        "keep_mask": keep,
        "dense_probability_sum_error": dense_sum_error.reshape(1),
        "pruned_probability_sum_error": pruned_sum_error.reshape(1),
    }


def _source_masses(values: torch.Tensor, slices: dict[str, list[int]]) -> dict[str, float]:
    result = {}
    for name in ("video", "dream", "action"):
        start, end = (int(x) for x in slices[name])
        result[name] = float(values[start:end].sum().item())
    return result


def threshold_record_summary(
    record: dict[str, Any], reconstructed: dict[str, torch.Tensor]
) -> dict[str, Any]:
    keep = reconstructed["keep_mask"]
    dense_mass = _source_masses(reconstructed["dense_all_by_token"], record["slices"])
    pruned_mass = _source_masses(reconstructed["pruned_all_by_token"], record["slices"])
    return {
        "layer": int(record["layer"]),
        "denoise_step": int(record.get("denoise_step", -1)),
        "alpha": float(record["threshold"]["alpha_current"]),
        "num_dream_tokens": int(keep.numel()),
        "kept_dream_tokens": int(keep.sum().item()),
        "pruned_dream_tokens": int((~keep).sum().item()),
        "keep_ratio": float(keep.float().mean().item()),
        "dense_mass_video": dense_mass["video"],
        "dense_mass_dream": dense_mass["dream"],
        "dense_mass_action": dense_mass["action"],
        "pruned_mass_video": pruned_mass["video"],
        "pruned_mass_dream": pruned_mass["dream"],
        "pruned_mass_action": pruned_mass["action"],
        "dense_probability_sum_error": float(
            reconstructed["dense_probability_sum_error"].item()
        ),
        "pruned_probability_sum_error": float(
            reconstructed["pruned_probability_sum_error"].item()
        ),
    }


def _token_segments(metadata: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    if not metadata:
        return []
    labels = [f"{item['modality']}@+{item['future_offset']}" for item in metadata]
    segments = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            segments.append((start, index, labels[start]))
            start = index
    return segments


def save_threshold_record(
    *,
    record: dict[str, Any],
    token_metadata: list[dict[str, Any]],
    output_prefix: Path,
    save_headwise: bool = True,
) -> dict[str, Any]:
    """Save one layer/step record as dashboard, CSV, JSON, and NPZ."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    reconstructed = reconstruct_threshold_attention(record)
    summary = threshold_record_summary(record, reconstructed)
    score = reconstructed["score"].numpy()
    keep = reconstructed["keep_mask"].numpy()
    dense_token = reconstructed["dense_dream_by_token"].numpy()
    pruned_token = reconstructed["pruned_dream_by_token"].numpy()
    dense_matrix = reconstructed["dense_dream_by_action"].numpy()
    pruned_matrix = reconstructed["pruned_dream_by_action"].numpy()
    if len(token_metadata) != len(score):
        raise ValueError(
            f"Token metadata length {len(token_metadata)} does not match Dream length {len(score)}."
        )

    token_rows = []
    for index, metadata in enumerate(token_metadata):
        token_rows.append(
            {
                **metadata,
                "score": float(score[index]),
                "alpha": summary["alpha"],
                "kept": bool(keep[index]),
                "pruned": bool(not keep[index]),
                "dense_attention_probability": float(dense_token[index]),
                "pruned_attention_probability": float(pruned_token[index]),
            }
        )
    csv_path = output_prefix.with_suffix(".tokens.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(token_rows[0]))
        writer.writeheader()
        writer.writerows(token_rows)

    npz_payload = {
        "score": score,
        "keep_mask": keep,
        "dense_dream_by_action": dense_matrix,
        "pruned_dream_by_action": pruned_matrix,
        "dense_dream_by_token": dense_token,
        "pruned_dream_by_token": pruned_token,
        "dense_all_by_token": reconstructed["dense_all_by_token"].numpy(),
        "pruned_all_by_token": reconstructed["pruned_all_by_token"].numpy(),
    }
    if save_headwise:
        npz_payload["dense_dream_headwise"] = reconstructed[
            "dense_dream_headwise"
        ].numpy()
        npz_payload["pruned_dream_headwise"] = reconstructed[
            "pruned_dream_headwise"
        ].numpy()
    np.savez_compressed(output_prefix.with_suffix(".attention.npz"), **npz_payload)
    output_prefix.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    segments = _token_segments(token_metadata)
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), constrained_layout=True)
    colors = np.where(keep, "#4C78A8", "#E45756")
    axes[0, 0].bar(np.arange(len(score)), score, color=colors, width=1.0)
    axes[0, 0].axhline(summary["alpha"], color="black", linestyle="--", linewidth=1.2)
    axes[0, 0].set_title(
        f"Dream score and decision: K={summary['kept_dream_tokens']}/{len(score)}, "
        f"alpha={summary['alpha']:.3f}"
    )
    axes[0, 0].set_ylabel("score = P × N_valid")

    axes[0, 1].plot(dense_token, label="dense/original", color="#F58518")
    axes[0, 1].plot(pruned_token, label="after threshold", color="#4C78A8")
    axes[0, 1].scatter(
        np.flatnonzero(~keep), dense_token[~keep], marker="x", color="#E45756", s=16,
        label="pruned token (original mass)",
    )
    axes[0, 1].set_title("Action→Dream probability by token")
    axes[0, 1].set_ylabel("mixed-attention probability")
    axes[0, 1].legend(fontsize=8)

    vmax = max(float(np.quantile(dense_matrix, 0.995)), float(np.quantile(pruned_matrix, 0.995)), 1e-8)
    dense_image = axes[1, 0].imshow(dense_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    axes[1, 0].set_title("Dense Action-query × Dream-token probability (head mean)")
    axes[1, 0].set_ylabel("Action query index")
    fig.colorbar(dense_image, ax=axes[1, 0], fraction=0.025, pad=0.02)
    pruned_image = axes[1, 1].imshow(pruned_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    axes[1, 1].set_title("Pruned Action-query × Dream-token probability (head mean)")
    axes[1, 1].set_ylabel("Action query index")
    fig.colorbar(pruned_image, ax=axes[1, 1], fraction=0.025, pad=0.02)

    names = ["video", "dream", "action-self"]
    dense_values = [summary["dense_mass_video"], summary["dense_mass_dream"], summary["dense_mass_action"]]
    pruned_values = [summary["pruned_mass_video"], summary["pruned_mass_dream"], summary["pruned_mass_action"]]
    x = np.arange(3)
    axes[2, 0].bar(x - 0.18, dense_values, width=0.36, label="dense")
    axes[2, 0].bar(x + 0.18, pruned_values, width=0.36, label="after threshold")
    axes[2, 0].set_xticks(x, names)
    axes[2, 0].set_ylim(0, 1)
    axes[2, 0].set_ylabel("mixed-attention mass")
    axes[2, 0].set_title("Source attention mass")
    axes[2, 0].legend()

    group_labels = []
    group_keep = []
    group_mass = []
    for start, end, label in segments:
        group_labels.append(label)
        group_keep.append(float(keep[start:end].mean()))
        denominator = float(dense_token[start:end].sum())
        group_mass.append(
            0.0 if denominator <= 0 else float(dense_token[start:end][keep[start:end]].sum()) / denominator
        )
    y = np.arange(len(group_labels))
    axes[2, 1].barh(y - 0.18, group_keep, height=0.36, label="token keep ratio")
    axes[2, 1].barh(y + 0.18, group_mass, height=0.36, label="original mass retained")
    axes[2, 1].set_yticks(y, group_labels)
    axes[2, 1].set_xlim(0, 1)
    axes[2, 1].set_title("Retention by modality and future horizon")
    axes[2, 1].legend(fontsize=8)

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]):
        for start, _, _ in segments[1:]:
            axis.axvline(start - 0.5, color="white" if axis in (axes[1, 0], axes[1, 1]) else "0.7", linewidth=0.6)
        axis.set_xlabel("Dream token index")
    fig.suptitle(
        f"Layer {summary['layer']} · denoise step {summary['denoise_step']} · "
        f"Dream mass {summary['dense_mass_dream']:.3f}→{summary['pruned_mass_dream']:.3f}",
        fontsize=14,
    )
    fig.savefig(output_prefix.with_suffix(".dashboard.png"), dpi=160)
    plt.close(fig)
    return summary

