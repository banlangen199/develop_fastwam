from __future__ import annotations

import csv

import numpy as np
import torch

from action_dream_alpha.threshold_visualization import (
    reconstruct_threshold_attention,
    save_threshold_record,
    threshold_record_summary,
)


def _record() -> dict:
    probs = torch.tensor(
        [[[[0.20, 0.10, 0.15, 0.05, 0.30, 0.20],
           [0.10, 0.20, 0.10, 0.10, 0.25, 0.25]],
          [[0.15, 0.15, 0.20, 0.10, 0.20, 0.20],
           [0.20, 0.10, 0.05, 0.15, 0.25, 0.25]]]]
    )
    return {
        "layer": 3,
        "denoise_step": 4,
        "probs": probs,
        "slices": {"video": [0, 2], "dream": [2, 4], "action": [4, 6]},
        "threshold": {
            "alpha_current": 1.0,
            "score": torch.tensor([[1.2, 0.7]]),
            "keep_mask": torch.tensor([[True, False]]),
        },
    }


def test_reconstruct_masks_only_dream_and_renormalizes_full_mixed_distribution() -> None:
    record = _record()
    result = reconstruct_threshold_attention(record)
    dense = result["dense_probs"]
    pruned = result["pruned_probs"]
    torch.testing.assert_close(dense.sum(dim=-1), torch.ones_like(dense.sum(dim=-1)))
    torch.testing.assert_close(pruned.sum(dim=-1), torch.ones_like(pruned.sum(dim=-1)))
    assert bool((pruned[..., 3] == 0).all())
    assert bool((pruned[..., 2] > 0).all())
    ratio_before = dense[..., 0] / dense[..., 4]
    ratio_after = pruned[..., 0] / pruned[..., 4]
    torch.testing.assert_close(ratio_after, ratio_before)
    summary = threshold_record_summary(record, result)
    assert summary["kept_dream_tokens"] == 1
    assert summary["pruned_dream_tokens"] == 1
    assert summary["dense_probability_sum_error"] < 1e-6
    assert summary["pruned_probability_sum_error"] < 1e-6


def test_save_record_exposes_token_identity_decision_and_attention(tmp_path) -> None:
    metadata = [
        {"token_index": 0, "modality": "dyn", "future_offset": 16, "slot_id": 0},
        {"token_index": 1, "modality": "depth", "future_offset": 32, "slot_id": 0},
    ]
    prefix = tmp_path / "layer_03_step_04"
    summary = save_threshold_record(
        record=_record(), token_metadata=metadata, output_prefix=prefix
    )
    assert summary["kept_dream_tokens"] == 1
    assert prefix.with_suffix(".dashboard.png").exists()
    assert prefix.with_suffix(".summary.json").exists()
    rows = list(csv.DictReader(prefix.with_suffix(".tokens.csv").open()))
    assert [row["kept"] for row in rows] == ["True", "False"]
    payload = np.load(prefix.with_suffix(".attention.npz"))
    assert payload["dense_dream_headwise"].shape == (2, 2, 2)
    assert payload["pruned_dream_by_action"].shape == (2, 2)

