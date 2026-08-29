from __future__ import annotations

import copy
import inspect

import pytest
import torch
import torch.nn as nn

from fastwam.models.wan22.action_dream_threshold import (
    ActionDreamThresholdConfig,
    ActionDreamThresholdMoT,
    ThresholdDreamFastWAM,
    rope_pair_uniform_channel_indices,
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


def _cached_action_forward(mot, args):
    embeds, freqs, t_mod, context, mask = args
    video_len = int(embeds["video"].shape[1])
    dream_len = int(embeds["dream"].shape[1])
    context_len = video_len + dream_len
    prefill = mot.prefill_video_dream_cache(
        video_tokens=embeds["video"],
        dream_tokens=embeds["dream"],
        video_freqs=freqs["video"],
        dream_freqs=freqs["dream"],
        video_t_mod=t_mod["video"],
        dream_t_mod=t_mod["dream"],
        video_context_payload=context["video"],
        dream_context_payload=context["dream"],
        context_attention_mask=mask[:context_len, :context_len],
    )
    action = mot.forward_action_with_context_cache(
        action_tokens=embeds["action"],
        action_freqs=freqs["action"],
        action_t_mod=t_mod["action"],
        action_context_payload=context["action"],
        context_kv_cache=prefill["kv_cache"],
        attention_mask=mask,
        video_seq_len=video_len,
        dream_seq_len=dream_len,
    )
    return prefill, action


def test_video_dream_cache_matches_full_dense_mot():
    mot = MoT(make_mixtures(), mot_checkpoint_mixed_attn=False).eval()
    args = make_inputs(batch_size=1)
    full = mot(args[0], args[4], args[1], args[3], args[2])
    prefill, action = _cached_action_forward(mot, args)
    torch.testing.assert_close(prefill["tokens"]["video"], full["video"], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(prefill["tokens"]["dream"], full["dream"], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(action, full["action"], rtol=2e-5, atol=2e-6)


def test_video_dream_cache_preserves_action_dream_threshold():
    mixtures = make_mixtures()
    dense = MoT(copy.deepcopy(mixtures), mot_checkpoint_mixed_attn=False).eval()
    threshold = make_threshold_mot(copy.deepcopy(mixtures), alpha=1.0e6).eval()
    threshold.load_state_dict(dense.state_dict(), strict=True)
    args = make_inputs(batch_size=1)
    full_threshold = threshold(args[0], args[4], args[1], args[3], args[2])["action"]
    _, cached_threshold = _cached_action_forward(threshold, args)
    _, cached_dense = _cached_action_forward(dense, args)
    torch.testing.assert_close(cached_threshold, full_threshold, rtol=2e-5, atol=2e-6)
    assert not torch.equal(cached_threshold, cached_dense)


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


def test_rope_pair_uniform_channels_are_deterministic_and_pair_complete():
    channels = rope_pair_uniform_channel_indices(128, 16)
    assert channels.tolist() == [
        0, 1, 16, 17, 32, 33, 48, 49,
        64, 65, 80, 81, 96, 97, 112, 113,
    ]
    assert channels.numel() == 16
    assert torch.equal(channels[1::2], channels[::2] + 1)
    assert bool((channels[::2] % 2 == 0).all())


@pytest.mark.parametrize("proxy_head_dim", [0, 3])
def test_proxy_head_dim_validation_rejects_non_positive_or_odd(proxy_head_dim):
    with pytest.raises(ValueError, match="proxy_head_dim"):
        ActionDreamThresholdConfig.from_dict({"proxy_head_dim": proxy_head_dim})


def test_proxy_head_dim_must_fit_the_real_attention_head():
    with pytest.raises(ValueError, match="proxy_head_dim"):
        make_threshold_mot(proxy_analysis_enabled=True, proxy_head_dim=16)


def test_proxy_analysis_does_not_change_real_action_output():
    torch.manual_seed(17)
    q = torch.randn(2, 2, 8)
    k = torch.randn(2, 6, 8)
    v = torch.randn(2, 6, 8)
    mask = torch.ones(6, 6, dtype=torch.bool)
    mask[4, 0] = False
    teacher_only = make_threshold_mot(proxy_analysis_enabled=False)
    with_proxy = make_threshold_mot(proxy_analysis_enabled=True, proxy_head_dim=2)
    teacher_result = direct_threshold_call(teacher_only, q, k, v, alpha=0.9, mask=mask)
    proxy_result = direct_threshold_call(with_proxy, q, k, v, alpha=0.9, mask=mask)
    torch.testing.assert_close(proxy_result[0], teacher_result[0], rtol=0, atol=0)
    torch.testing.assert_close(proxy_result[1], teacher_result[1], rtol=0, atol=0)
    assert torch.equal(proxy_result[2], teacher_result[2])
    assert teacher_result[7] is None
    assert proxy_result[7] is not None


def test_proxy_analysis_does_not_change_end_to_end_mot_outputs():
    mixtures = make_mixtures()
    teacher = make_threshold_mot(
        copy.deepcopy(mixtures),
        alpha=0.9,
        proxy_analysis_enabled=False,
    ).eval()
    proxy = make_threshold_mot(
        copy.deepcopy(mixtures),
        alpha=0.9,
        proxy_analysis_enabled=True,
        proxy_head_dim=2,
        save_detailed_tensors=True,
    ).eval()
    proxy.load_state_dict(teacher.state_dict(), strict=True)
    args = make_inputs(batch_size=1)
    teacher_out = teacher(args[0], args[4], args[1], args[3], args[2])
    proxy_result = proxy(
        args[0],
        args[4],
        args[1],
        args[3],
        args[2],
        return_threshold_statistics=True,
    )
    for name in ("video", "dream", "action"):
        torch.testing.assert_close(
            proxy_result["tokens"][name],
            teacher_out[name],
            rtol=0,
            atol=0,
        )
    for record in proxy_result["action_dream_threshold"]:
        assert record["full_score"].shape == (1, 4)
        assert record["proxy_score"].shape == (1, 4)
        assert record["full_keep_mask"].shape == (1, 4)
        assert record["proxy_keep_mask"].shape == (1, 4)


def test_full_dimension_proxy_matches_teacher_score_and_mask():
    torch.manual_seed(19)
    mot = make_threshold_mot(
        proxy_analysis_enabled=True,
        proxy_head_dim=4,
        save_detailed_tensors=True,
    )
    q = torch.randn(2, 2, 8)
    k = torch.randn(2, 6, 8)
    v = torch.randn(2, 6, 8)
    _, score, keep, *_, proxy = direct_threshold_call(mot, q, k, v, alpha=1.0)
    torch.testing.assert_close(proxy["proxy_score"], score, rtol=0, atol=0)
    assert torch.equal(proxy["proxy_keep_mask"], keep)
    assert bool((proxy["teacher_recall"] == 1).all())
    assert bool((proxy["teacher_precision"] == 1).all())
    assert bool((proxy["teacher_jaccard"] == 1).all())


def test_proxy_uses_full_joint_key_space_and_proxy_scale():
    torch.manual_seed(23)
    mot = make_threshold_mot(proxy_analysis_enabled=True, proxy_head_dim=2)
    q = torch.randn(1, 2, 8)
    k = torch.randn(1, 6, 8)
    v = torch.randn(1, 6, 8)
    mask = torch.ones(6, 6, dtype=torch.bool)
    mask[4, 0] = False
    result = direct_threshold_call(mot, q, k, v, alpha=0.7, mask=mask)
    proxy = result[7]
    assert proxy["proxy_score"].shape == (1, 2)
    assert proxy["proxy_keep_mask"].shape == (1, 2)

    qh = q.reshape(1, 2, 2, 4).transpose(1, 2)[..., :2]
    kh = k.reshape(1, 6, 2, 4).transpose(1, 2)[..., :2]
    allowed = mask[4:6].view(1, 1, 2, 6).expand(1, 2, 2, 6)
    logits = qh.float() @ kh.float().transpose(-2, -1) * (2**-0.5)
    probs = torch.softmax(logits.masked_fill(~allowed, -torch.inf), dim=-1)
    assert probs[..., :2].sum() > 0
    assert probs[..., 2:4].sum() > 0
    assert probs[..., 4:6].sum() > 0
    n_valid = allowed.sum(dim=-1)[:, 0].float()
    expected = (probs[..., 2:4] * n_valid[:, None, :, None]).mean(dim=1).amax(dim=1)
    torch.testing.assert_close(proxy["proxy_score"], expected)


def test_proxy_minimum_keep_and_empty_mask_metrics_are_finite():
    torch.manual_seed(29)
    q = torch.randn(1, 2, 8)
    k = torch.randn(1, 6, 8)
    v = torch.randn(1, 6, 8)

    minimum = make_threshold_mot(
        proxy_analysis_enabled=True,
        proxy_head_dim=2,
        min_keep_dream_tokens=1,
    )
    minimum_result = direct_threshold_call(minimum, q, k, v, alpha=1.0e6)
    assert minimum_result[2].sum().item() == 1
    assert minimum_result[7]["proxy_keep_mask"].sum().item() == 1

    empty = make_threshold_mot(proxy_analysis_enabled=True, proxy_head_dim=2)
    proxy = direct_threshold_call(empty, q, k, v, alpha=1.0e6)[7]
    assert proxy["full_k"].item() == 0
    assert proxy["proxy_k"].item() == 0
    for key in (
        "teacher_recall",
        "teacher_precision",
        "teacher_jaccard",
        "mask_agreement",
        "score_mae",
        "score_rmse",
        "proxy_full_mass_retention",
    ):
        assert torch.isfinite(proxy[key]).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_proxy_analysis_has_no_nan_in_supported_cpu_dtypes(dtype):
    mot = make_threshold_mot(proxy_analysis_enabled=True, proxy_head_dim=2)
    q = torch.randn(1, 2, 8, dtype=dtype)
    k = torch.randn(1, 6, 8, dtype=dtype)
    v = torch.randn(1, 6, 8, dtype=dtype)
    result = direct_threshold_call(mot, q, k, v, alpha=1.0)
    assert result[0].dtype == dtype
    assert torch.isfinite(result[0]).all()
    proxy = result[7]
    for value in proxy.values():
        if isinstance(value, torch.Tensor) and value.dtype is not torch.bool:
            assert torch.isfinite(value).all()


def test_proxy_analysis_supports_training_gradient_checkpointing():
    mot = ActionDreamThresholdMoT(
        mixtures=make_mixtures(num_layers=1),
        mot_checkpoint_mixed_attn=True,
        action_dream_threshold={
            "enabled": True,
            "alpha": 0.8,
            "warmup_ratio": 0.0,
            "proxy_analysis_enabled": True,
            "proxy_head_dim": 2,
        },
    ).train()
    args = make_inputs(batch_size=1)
    result = mot(args[0], args[4], args[1], args[3], args[2])
    result["action"].square().mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in mot.mixtures["action"].parameters()
    )
    assert len(mot.last_threshold_statistics) == 1
    assert mot.last_threshold_statistics[0]["proxy_head_dim"] == 2


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
