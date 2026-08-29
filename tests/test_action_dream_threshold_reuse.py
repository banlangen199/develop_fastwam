from __future__ import annotations

import torch
import torch.nn as nn

from fastwam.models.wan22.action_dream_threshold_reuse import (
    ActionDreamThresholdReuseMoT,
)
from fastwam.models.wan22.wan_video_dit import DiTBlock


class _TinyExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 2
        self.attn_head_dim = 4
        self.use_gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=8,
                    attn_head_dim=self.attn_head_dim,
                    num_heads=self.num_heads,
                    ffn_dim=16,
                )
                for _ in range(2)
            ]
        )


def _make_mixtures() -> dict[str, _TinyExpert]:
    return {name: _TinyExpert() for name in ("video", "dream", "action")}


def _make_mot() -> ActionDreamThresholdReuseMoT:
    return ActionDreamThresholdReuseMoT(
        mixtures=_make_mixtures(),
        mot_checkpoint_mixed_attn=False,
        action_dream_threshold={
            "enabled": True,
            "alpha": 1.0,
            "warmup_ratio": 0.0,
            "head_reduce": "mean",
            "action_query_reduce": "max",
            "min_keep_dream_tokens": 0,
            "detach_selection_score": True,
            "log_statistics": False,
            "save_detailed_tensors": False,
        },
        action_dream_threshold_reuse={"refresh_steps": [0, 2]},
    ).eval()


def test_reuse_mask_is_refreshed_only_at_configured_steps() -> None:
    mot = _make_mot()
    torch.manual_seed(3)
    q = torch.randn(1, 2, 8)
    k = torch.randn(1, 9, 8)
    v = torch.randn(1, 9, 8)
    mask = torch.ones(9, 9, dtype=torch.bool)
    context_slices = {"video": slice(0, 3), "dream": slice(3, 7)}
    action_slice = slice(7, 9)

    mot.begin_reuse_inference(3)
    try:
        mot._action_attention_with_context_cache(
            q_action=q, k_all=k, v_all=v, attention_mask=mask,
            action_slice=action_slice, context_slices=context_slices, layer_idx=0,
        )
        first = mot._reuse_layer_masks[0]
        assert first is not None
        mot._reuse_step_index = 1
        compact = mot._action_attention_with_context_cache(
            q_action=q, k_all=k, v_all=v, attention_mask=mask,
            action_slice=action_slice, context_slices=context_slices, layer_idx=0,
        )
        assert compact.shape == q.shape
        assert mot._reuse_layer_masks[0] is first
        mot._reuse_step_index = 2
        mot._action_attention_with_context_cache(
            q_action=q, k_all=k, v_all=v, attention_mask=mask,
            action_slice=action_slice, context_slices=context_slices, layer_idx=0,
        )
        assert mot._reuse_layer_masks[0] is not first
    finally:
        mot.end_reuse_inference()


def test_step_zero_remains_active_until_its_action_forward_finishes() -> None:
    mot = _make_mot()
    mot.begin_reuse_inference(3)
    try:
        assert mot._reuse_step_index == 0
        mot.advance_reuse_inference_step()
        assert mot._reuse_step_index == 1
    finally:
        mot.end_reuse_inference()


def test_compact_attention_matches_attention_with_the_same_pruned_mask() -> None:
    mot = _make_mot()
    torch.manual_seed(4)
    q = torch.randn(1, 2, 8)
    k = torch.randn(1, 9, 8)
    v = torch.randn(1, 9, 8)
    mask = torch.ones(9, 9, dtype=torch.bool)
    action_slice = slice(7, 9)
    context_slices = {"video": slice(0, 3), "dream": slice(3, 7)}
    keep = torch.tensor([[True, False, True, False]])
    packed = mot._pack_keep_mask(keep)
    compact = mot._compact_action_attention(
        q_action=q, k_all=k, v_all=v, attention_mask=mask,
        action_slice=action_slice, context_slices=context_slices, packed_keep=packed,
    )
    full_mask = mask.clone()
    full_mask[action_slice, context_slices["dream"]] &= keep[0]
    reference = mot._threshold_action_attention(
        q_action=q, k_all=k, v_all=v, attention_mask=full_mask,
        action_slice=action_slice, dream_slice=context_slices["dream"], alpha=0.0,
    )[0]
    torch.testing.assert_close(compact, reference, rtol=2e-5, atol=2e-6)
