from __future__ import annotations

import copy
import inspect

import pytest
import torch
import torch.nn as nn

from fastwam.models.wan22.action_dream_threshold import (
    ActionDreamThresholdMoT,
    ThresholdDreamFastWAM,
)
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import DiTBlock, precompute_freqs_cis


class TinyExpert(nn.Module):
    def __init__(self, *, num_layers: int = 2, hidden_dim: int = 8, num_heads: int = 2, head_dim: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_dim = head_dim
        self.use_gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=head_dim,
                    num_heads=num_heads,
                    ffn_dim=hidden_dim * 2,
                )
                for _ in range(num_layers)
            ]
        )


def make_mixtures(num_layers: int = 2) -> dict[str, TinyExpert]:
    return {
        "video": TinyExpert(num_layers=num_layers),
        "dream": TinyExpert(num_layers=num_layers),
        "action": TinyExpert(num_layers=num_layers),
    }


def make_inputs(batch_size: int = 2):
    torch.manual_seed(7)
    lengths = {"video": 3, "dream": 4, "action": 2}
    embeds = {name: torch.randn(batch_size, length, 8) for name, length in lengths.items()}
    freqs = {
        name: precompute_freqs_cis(4, end=length).view(length, 1, -1)
        for name, length in lengths.items()
    }
    t_mod = {name: torch.zeros(batch_size, 6, 8) for name in lengths}
    context = {name: None for name in lengths}
    total = sum(lengths.values())
    mask = torch.zeros(total, total, dtype=torch.bool)
    video = slice(0, 3)
    dream = slice(3, 7)
    action = slice(7, 9)
    mask[video, video] = True
    mask[dream, :3] = True
    mask[dream, dream] = True
    mask[action, :] = True
    return embeds, freqs, t_mod, context, mask


def make_threshold_mot(mixtures=None, **overrides):
    cfg = {
        "enabled": True,
        "alpha": 1.0,
        "warmup_ratio": 0.0,
        "head_reduce": "mean",
        "action_query_reduce": "max",
        "min_keep_dream_tokens": 0,
        "detach_selection_score": True,
        "log_statistics": True,
        "save_detailed_tensors": False,
    }
    cfg.update(overrides)
    return ActionDreamThresholdMoT(
        mixtures=make_mixtures() if mixtures is None else mixtures,
        mot_checkpoint_mixed_attn=False,
        action_dream_threshold=cfg,
    )


def direct_threshold_call(mot, q, k, v, *, alpha=1.0, mask=None):
    batch_size = q.shape[0]
    if mask is None:
        mask = torch.ones(6, 6, dtype=torch.bool)
    return mot._threshold_action_attention(
        q_action=q,
        k_all=k,
        v_all=v,
        attention_mask=mask,
        action_slice=slice(4, 6),
        dream_slice=slice(2, 4),
        alpha=alpha,
    )


def test_enabled_false_and_alpha_zero_match_original_dense_output():
    mixtures = make_mixtures()
    dense = MoT(copy.deepcopy(mixtures), mot_checkpoint_mixed_attn=False).eval()
    disabled = make_threshold_mot(copy.deepcopy(mixtures), enabled=False).eval()
    alpha_zero = make_threshold_mot(copy.deepcopy(mixtures), alpha=0.0).eval()
    disabled.load_state_dict(dense.state_dict(), strict=True)
    alpha_zero.load_state_dict(dense.state_dict(), strict=True)
    args = make_inputs()
    dense_out = dense(args[0], args[4], args[1], args[3], args[2])
    disabled_out = disabled(args[0], args[4], args[1], args[3], args[2])
    zero_out = alpha_zero(args[0], args[4], args[1], args[3], args[2])
    for name in ("video", "dream", "action"):
        torch.testing.assert_close(disabled_out[name], dense_out[name], rtol=0, atol=0)
        torch.testing.assert_close(zero_out[name], dense_out[name], rtol=0, atol=0)


def test_dense_softmax_uses_all_valid_sources_and_sums_to_one():
    mot = make_threshold_mot()
    torch.manual_seed(1)
    q = torch.randn(1, 2, 8)
    k = torch.randn(1, 6, 8)
    v = torch.randn(1, 6, 8)
    mask = torch.ones(6, 6, dtype=torch.bool)
    mask[4, 0] = False
    result = direct_threshold_call(mot, q, k, v, alpha=1.0, mask=mask)
    score, n_valid = result[1], result[3]

    qh = q.reshape(1, 2, 2, 4).transpose(1, 2)
    kh = k.reshape(1, 6, 2, 4).transpose(1, 2)
    logits = qh.float() @ kh.float().transpose(-2, -1) * (4**-0.5)
    allowed = mask[4:6].view(1, 1, 2, 6)
    probs = torch.softmax(logits.masked_fill(~allowed, -torch.inf), dim=-1)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones_like(probs[..., 0]))
    assert probs[..., 0:2].sum() > 0  # Video
    assert probs[..., 2:4].sum() > 0  # Dream
    assert probs[..., 4:6].sum() > 0  # Action-self
    torch.testing.assert_close(n_valid, torch.tensor([[5.0, 6.0]]))
    expected_score = (probs[..., 2:4] * n_valid[:, None, :, None]).mean(dim=1).amax(dim=1)
    assert score.shape == (1, 2)
    torch.testing.assert_close(score, expected_score)


def test_aggregation_is_head_mean_then_action_query_max_and_alpha_zero_keeps_all():
    mot = make_threshold_mot()
    torch.manual_seed(3)
    q = torch.randn(2, 2, 8)
    k = torch.randn(2, 6, 8)
    v = torch.randn(2, 6, 8)
    output, score, keep, *_ = direct_threshold_call(mot, q, k, v, alpha=0.0)
    assert output.shape == (2, 2, 8)
    assert score.shape == (2, 2)
    assert bool(keep.all())
    assert not score.requires_grad


def test_different_samples_and_layers_can_have_different_k():
    mot = make_threshold_mot()
    v = torch.randn(2, 6, 8)
    mask = torch.ones(6, 6, dtype=torch.bool)
    # Sample zero strongly matches Dream key 0; sample one strongly matches Dream key 1.
    q1 = torch.zeros(2, 2, 8)
    k1 = torch.zeros(2, 6, 8)
    q1[0, :, 0] = 5
    k1[0, 2, 0] = 5
    q1[1, :, 1] = 5
    k1[1, 3, 1] = 2
    _, _, keep1, *_ = direct_threshold_call(mot, q1, k1, v, alpha=1.0, mask=mask)

    q2 = q1.clone()
    k2 = k1.clone()
    k2[:, 2:4] = 0
    _, _, keep2, *_ = direct_threshold_call(mot, q2, k2, v, alpha=1.0, mask=mask)
    assert not torch.equal(keep1.sum(dim=-1), keep2.sum(dim=-1))
    assert keep1[0].sum() != keep1[1].sum() or not torch.equal(keep1[0], keep1[1])


def test_forward_layers_produce_different_k_from_their_own_action_attention():
    mixtures = make_mixtures(num_layers=2)
    with torch.no_grad():
        for expert in mixtures.values():
            for block in expert.blocks:
                block.self_attn.q.weight.zero_()
                block.self_attn.q.bias.zero_()
                block.self_attn.k.weight.zero_()
                block.self_attn.k.bias.zero_()
        # In layer one, every Action head matches every Dream head while Video
        # and Action-self keys stay at zero. Layer zero remains uniform.
        mixtures["action"].blocks[1].self_attn.q.bias[[0, 4]] = 10.0
        mixtures["dream"].blocks[1].self_attn.k.bias[[0, 4]] = 10.0
    mot = make_threshold_mot(mixtures, alpha=1.1).eval()
    args = make_inputs(batch_size=1)
    result = mot(
        args[0], args[4], args[1], args[3], args[2], return_threshold_statistics=True
    )
    records = result["action_dream_threshold"]
    assert records[0]["k_max"] == 0
    assert records[1]["k_min"] > 0


def test_k_zero_is_stable_and_only_action_to_dream_changes():
    mixtures = make_mixtures()
    dense = MoT(copy.deepcopy(mixtures), mot_checkpoint_mixed_attn=False).eval()
    threshold = make_threshold_mot(copy.deepcopy(mixtures), alpha=1.0e6).eval()
    threshold.load_state_dict(dense.state_dict(), strict=True)
    args = make_inputs(batch_size=1)
    dense_out = dense(args[0], args[4], args[1], args[3], args[2])
    pruned = threshold(
        args[0], args[4], args[1], args[3], args[2], return_threshold_statistics=True
    )
    assert all(record["k_max"] == 0 for record in pruned["action_dream_threshold"])
    assert torch.isfinite(pruned["tokens"]["action"]).all()
    torch.testing.assert_close(pruned["tokens"]["video"], dense_out["video"], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(pruned["tokens"]["dream"], dense_out["dream"], rtol=2e-5, atol=2e-6)
    assert not torch.equal(pruned["tokens"]["action"], dense_out["action"])


def test_hard_selection_detaches_but_pruned_attention_backpropagates():
    mot = make_threshold_mot()
    q = torch.randn(1, 2, 8, requires_grad=True)
    k = torch.randn(1, 6, 8, requires_grad=True)
    v = torch.randn(1, 6, 8, requires_grad=True)
    output, score, keep, *_ = direct_threshold_call(mot, q, k, v, alpha=1.0)
    assert not score.requires_grad
    assert not keep.requires_grad
    output.square().mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_frozen_video_and_dream_have_no_grad_while_action_expert_does():
    mot = make_threshold_mot(alpha=1.0).train()
    mot.mixtures["video"].requires_grad_(False)
    mot.mixtures["dream"].requires_grad_(False)
    mot.mixtures["action"].requires_grad_(True)
    args = make_inputs(batch_size=1)
    output = mot(args[0], args[4], args[1], args[3], args[2])
    output["action"].square().mean().backward()
    assert not any(parameter.grad is not None for parameter in mot.mixtures["video"].parameters())
    assert not any(parameter.grad is not None for parameter in mot.mixtures["dream"].parameters())
    assert any(parameter.grad is not None for parameter in mot.mixtures["action"].parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_threshold_attention_has_no_nan_in_supported_cpu_dtypes(dtype):
    mot = make_threshold_mot()
    q = torch.randn(1, 2, 8, dtype=dtype)
    k = torch.randn(1, 6, 8, dtype=dtype)
    v = torch.randn(1, 6, 8, dtype=dtype)
    output = direct_threshold_call(mot, q, k, v, alpha=1.0e6)[0]
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


def test_old_mot_checkpoint_keys_are_unchanged_and_no_router_parameters_exist():
    mixtures = make_mixtures()
    dense = MoT(copy.deepcopy(mixtures), mot_checkpoint_mixed_attn=False)
    threshold = make_threshold_mot(copy.deepcopy(mixtures))
    assert set(dense.state_dict()) == set(threshold.state_dict())
    threshold.load_state_dict(dense.state_dict(), strict=True)
    parameter_names = list(dict(threshold.named_parameters()))
    assert not any("router" in name or "threshold" in name for name in parameter_names)
    source = inspect.getsource(ActionDreamThresholdMoT)
    assert "torch.topk" not in source


def test_global_alpha_is_shared_by_every_layer_and_warmup_reaches_target():
    mot = make_threshold_mot(alpha=1.0, warmup_ratio=0.1)
    mot.train()
    mot.set_training_progress(global_step=5, total_steps=100)
    assert mot.current_alpha() == pytest.approx(0.5)
    args = make_inputs(batch_size=1)
    result = mot(
        args[0], args[4], args[1], args[3], args[2], return_threshold_statistics=True
    )
    assert {record["alpha_current"] for record in result["action_dream_threshold"]} == {0.5}
    mot.eval()
    assert mot.current_alpha() == 1.0


def test_training_at_completed_warmup_matches_inference_threshold_behavior():
    mot = make_threshold_mot(alpha=1.0, warmup_ratio=0.1)
    args = make_inputs(batch_size=1)
    mot.train()
    mot.set_training_progress(global_step=10, total_steps=100)
    train_out = mot(args[0], args[4], args[1], args[3], args[2])
    mot.eval()
    eval_out = mot(args[0], args[4], args[1], args[3], args[2])
    for name in ("video", "dream", "action"):
        torch.testing.assert_close(train_out[name], eval_out[name], rtol=2e-5, atol=2e-6)


def test_action_only_freeze_policy_selects_only_action_parameters():
    model = object.__new__(ThresholdDreamFastWAM)
    nn.Module.__init__(model)
    model.finetune_action_only = True
    model.video_expert = nn.Linear(3, 3)
    model.dream_expert = nn.Linear(3, 3)
    model.action_expert = nn.Linear(3, 3)
    model.proprio_encoder = nn.Linear(3, 3)
    model.mot = nn.Module()
    params = model.configure_trainable_parameters(freeze_video_expert=True)
    assert params == list(model.action_expert.parameters())
    assert all(parameter.requires_grad for parameter in model.action_expert.parameters())
    assert not any(parameter.requires_grad for parameter in model.video_expert.parameters())
    assert not any(parameter.requires_grad for parameter in model.dream_expert.parameters())
    assert not any(parameter.requires_grad for parameter in model.proprio_encoder.parameters())
