"""Unit tests for the RoutedWAM additions.

Everything here runs on CPU with randomly initialised tiny modules: no dataset,
no Wan2.2 download and no checkpoint is required, so the tests are runnable in
any environment that can import the package.

The properties under test are the ones that would silently corrupt results if
they broke:

* routing off must be *exactly* the shipped dense computation;
* a gate of 1 must reproduce the dense attention;
* ``threshold`` mode must agree with the already-validated implementation in
  ``action_dream_threshold``;
* the split Video-once / Dream / Action-cached path must equal the joint
  forward, because that equivalence is what lets us train through the
  deployment computation;
* the interface loss must vanish when teacher and student agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.models.wan22.action_dream_threshold.model import ActionDreamThresholdMoT  # noqa: E402
from fastwam.models.wan22.dream_fastwam.dream_query_expert import DreamQueryExpert  # noqa: E402
from fastwam.models.wan22.dream_fastwam.model import DreamFastWAM  # noqa: E402
from fastwam.models.wan22.mot import MoT  # noqa: E402
from fastwam.models.wan22.routed_wam import (  # noqa: E402
    GenerativeDreamExpert,
    ImaginationRouter,
    InterfaceDistillConfig,
    InterfaceDistiller,
    RoutedMoT,
    RoutedWAM,
    RouterConfig,
    build_group_ids,
)
from fastwam.models.wan22.schedulers.scheduler_continuous import (  # noqa: E402
    WanContinuousFlowMatchScheduler,
)
from fastwam.models.wan22.wan_video_dit import DiTBlock, WanVideoDiT, precompute_freqs_cis  # noqa: E402


HIDDEN = 16
FFN = 32
HEADS = 2
HEAD_DIM = 8
LAYERS = 2
INNER = HEADS * HEAD_DIM
TOKENS_PER_FRAME = 4
NUM_FRAMES = 3
ACTION_TOKENS = 5


class TinyExpert(nn.Module):
    """Minimal stand-in exposing the attributes MoT reads off an expert."""

    def __init__(self, hidden_dim: int = HIDDEN):
        super().__init__()
        self.num_heads = HEADS
        self.attn_head_dim = HEAD_DIM
        self.hidden_dim = hidden_dim
        self.use_gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=HEAD_DIM,
                    num_heads=HEADS,
                    ffn_dim=FFN,
                    eps=1e-6,
                )
                for _ in range(LAYERS)
            ]
        )


def make_dream_expert(generative: bool = False) -> DreamQueryExpert:
    expert = DreamQueryExpert(
        text_dim=HIDDEN,
        freq_dim=8,
        eps=1e-6,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        dream_query=dict(
            modalities=["depth", "dino"],
            future_offsets=[0, 1],
            camera_token_split=[2, 2],
            n_depth=4,
            n_dino=4,
        ),
        dream_expert=dict(
            hidden_dim=HIDDEN,
            ffn_dim=FFN,
            num_layers=LAYERS,
            num_heads=HEADS,
            attn_head_dim=HEAD_DIM,
            mlp_ratio=2.0,
            share_blocks=False,
        ),
        dream_decoder=dict(
            decoder_dim=HIDDEN,
            decoder_ffn_dim=FFN,
            num_layers=1,
            num_heads=HEADS,
            attn_head_dim=HEAD_DIM,
            mlp_ratio=2.0,
            depth=dict(target_layout="token_feature", target_shape=[8, 3]),
            dino=dict(target_layout="grid_feature", target_shape=[2, 4, 5]),
        ),
    )
    if generative:
        GenerativeDreamExpert.promote(expert, {"enabled": True, "freq_dim": 8})
    return expert


def build_mask(video_seq_len: int, dream_expert: DreamQueryExpert, action_seq_len: int) -> torch.Tensor:
    """Use the shipped mask constructor rather than re-deriving it."""
    video_stub = object.__new__(WanVideoDiT)
    nn.Module.__init__(video_stub)
    video_stub.video_attention_mask_mode = "first_frame_causal"
    model_stub = object.__new__(DreamFastWAM)
    nn.Module.__init__(model_stub)
    model_stub.video_expert = video_stub
    model_stub.dream_expert = dream_expert
    return model_stub._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        dream_seq_len=dream_expert.num_dream_tokens,
        action_seq_len=action_seq_len,
        video_tokens_per_frame=TOKENS_PER_FRAME,
        device=torch.device("cpu"),
    )


def make_parts(seed: int = 0, generative: bool = False):
    torch.manual_seed(seed)
    dream_expert = make_dream_expert(generative=generative)
    mixtures = {
        "video": TinyExpert(),
        "dream": dream_expert,
        "action": TinyExpert(),
    }
    video_seq_len = TOKENS_PER_FRAME * NUM_FRAMES
    dream_seq_len = dream_expert.num_dream_tokens
    mask = build_mask(video_seq_len, dream_expert, ACTION_TOKENS)

    batch = 2
    freqs = precompute_freqs_cis(HEAD_DIM, end=256)
    embeds = {
        "video": torch.randn(batch, video_seq_len, HIDDEN),
        "dream": torch.randn(batch, dream_seq_len, HIDDEN),
        "action": torch.randn(batch, ACTION_TOKENS, HIDDEN),
    }
    freqs_all = {
        "video": freqs[:video_seq_len].view(video_seq_len, 1, -1),
        "dream": freqs[:dream_seq_len].view(dream_seq_len, 1, -1),
        "action": freqs[:ACTION_TOKENS].view(ACTION_TOKENS, 1, -1),
    }
    t_mod_all = {name: torch.randn(batch, 6, HIDDEN) for name in embeds}
    context_all = {name: None for name in embeds}
    return {
        "mixtures": mixtures,
        "embeds": embeds,
        "freqs": freqs_all,
        "t_mod": t_mod_all,
        "context": context_all,
        "mask": mask,
        "video_seq_len": video_seq_len,
        "dream_seq_len": dream_seq_len,
        "batch": batch,
        "dream_expert": dream_expert,
    }


def run_mot(mot, parts):
    return mot(
        embeds_all=parts["embeds"],
        attention_mask=parts["mask"],
        freqs_all=parts["freqs"],
        context_all=parts["context"],
        t_mod_all=parts["t_mod"],
    )


def make_router(parts, **overrides) -> ImaginationRouter:
    dream_expert = parts["dream_expert"]
    config = RouterConfig.from_dict(overrides)
    group_ids, group_names = build_group_ids(
        modalities=list(dream_expert.modalities),
        num_future_offsets=int(dream_expert.num_future_offsets),
        tokens_per_modality={
            name: int(getattr(dream_expert, f"n_{name}")) for name in dream_expert.modalities
        },
        granularity=config.group_granularity,
        camera_token_split=dream_expert.camera_token_split,
    )
    return ImaginationRouter(
        config=config,
        num_layers=LAYERS,
        inner_dim=INNER,
        num_dream_tokens=parts["dream_seq_len"],
        group_ids=group_ids,
        group_names=group_names,
    )

# --------------------------------------------------------------------- tests
def test_group_ids_follow_dream_token_layout():
    ids, names = build_group_ids(
        modalities=["depth", "dino"],
        num_future_offsets=2,
        tokens_per_modality={"depth": 4, "dino": 4},
        granularity="modality_horizon",
    )
    assert names == ["depth@t0", "depth@t1", "dino@t0", "dino@t1"]
    assert ids.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4

    ids, names = build_group_ids(
        modalities=["depth", "dino"],
        num_future_offsets=2,
        tokens_per_modality={"depth": 4, "dino": 4},
        granularity="modality",
    )
    assert names == ["depth", "dino"]
    assert ids.tolist() == [0] * 8 + [1] * 8


def test_routing_disabled_is_exactly_the_dense_computation():
    parts = make_parts()
    dense = MoT(mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False)
    routed = RoutedMoT(mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False, router=None)
    dense.eval()
    routed.eval()
    with torch.no_grad():
        expected = run_mot(dense, parts)
        actual = run_mot(routed, parts)
    for name in expected:
        assert torch.equal(expected[name], actual[name]), f"{name} diverged with routing disabled"


def test_gate_of_one_reproduces_dense_attention():
    parts = make_parts()
    dense = MoT(mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False)
    router = make_router(parts, mode="learned", debug_force_gate=1.0)
    routed = RoutedMoT(mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False, router=router)
    dense.eval()
    routed.eval()
    with torch.no_grad():
        expected = run_mot(dense, parts)
        actual = run_mot(routed, parts)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=1e-4, atol=1e-4)


def test_gate_of_zero_removes_all_dream_evidence():
    parts = make_parts()
    router = make_router(parts, mode="learned", debug_force_gate=0.0)
    routed = RoutedMoT(mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False, router=router)
    routed.eval()
    with torch.no_grad():
        with_dream = run_mot(routed, parts)
        perturbed = dict(parts)
        perturbed["embeds"] = dict(parts["embeds"])
        perturbed["embeds"]["dream"] = torch.randn_like(parts["embeds"]["dream"]) * 10.0
        without_dream = run_mot(routed, perturbed)
    # Action output must not depend on the dream tokens once the gate is closed.
    torch.testing.assert_close(
        with_dream["action"], without_dream["action"], rtol=1e-4, atol=1e-4
    )


def test_threshold_mode_matches_shipped_implementation():
    parts = make_parts()
    shipped = ActionDreamThresholdMoT(
        mixtures=parts["mixtures"],
        mot_checkpoint_mixed_attn=False,
        action_dream_threshold={"enabled": True, "alpha": 1.0, "warmup_ratio": 0.0},
    )
    router = make_router(parts, mode="threshold", alpha=1.0, warmup_ratio=0.0)
    routed = RoutedMoT(mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False, router=router)
    shipped.eval()
    routed.eval()
    with torch.no_grad():
        expected = run_mot(shipped, parts)
        actual = run_mot(routed, parts)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("router_mode", ["none", "learned"])
def test_split_prefill_equals_joint_forward(router_mode):
    """Video-once + Dream + cached Action must equal the joint MoT forward.

    This is the invariant that lets training run through the deployment
    computation. It holds because Video never reads Dream or Action, and Dream
    never reads Action, so their layer states are unaffected by being computed
    separately.
    """
    parts = make_parts()
    router = None if router_mode == "none" else make_router(parts, mode="learned", debug_force_gate=0.7)
    routed = RoutedMoT(mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False, router=router)
    routed.eval()

    video_seq_len = parts["video_seq_len"]
    dream_seq_len = parts["dream_seq_len"]
    context_seq_len = video_seq_len + dream_seq_len
    mask = parts["mask"]

    with torch.no_grad():
        joint = run_mot(routed, parts)

        video_kv = routed.prefill_video_cache(
            video_tokens=parts["embeds"]["video"],
            video_freqs=parts["freqs"]["video"],
            video_t_mod=parts["t_mod"]["video"],
            video_context_payload=None,
            video_attention_mask=mask[:video_seq_len, :video_seq_len],
        )
        dream_out = routed.forward_dream_with_video_cache(
            dream_tokens=parts["embeds"]["dream"],
            dream_freqs=parts["freqs"]["dream"],
            dream_t_mod=parts["t_mod"]["dream"],
            dream_context_payload=None,
            video_kv_cache=video_kv,
            context_attention_mask=mask[:context_seq_len, :context_seq_len],
            video_seq_len=video_seq_len,
        )
        action_out = routed.forward_action_with_context_cache(
            action_tokens=parts["embeds"]["action"],
            action_freqs=parts["freqs"]["action"],
            action_t_mod=parts["t_mod"]["action"],
            action_context_payload=None,
            context_kv_cache=routed.merge_context_cache(video_kv, dream_out["dream_kv"]),
            attention_mask=mask,
            video_seq_len=video_seq_len,
            dream_seq_len=dream_seq_len,
        )

    torch.testing.assert_close(joint["dream"], dream_out["tokens"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(joint["action"], action_out, rtol=1e-4, atol=1e-5)


def test_interface_loss_is_zero_for_an_identical_interface():
    parts = make_parts(generative=True)
    distiller = InterfaceDistiller(
        config=InterfaceDistillConfig.from_dict({"enabled": True, "teacher_steps": 2}),
        dream_expert=parts["dream_expert"],
        num_layers=LAYERS,
    )
    cache = [
        {"k": torch.randn(2, parts["dream_seq_len"], INNER), "v": torch.randn(2, parts["dream_seq_len"], INNER)}
        for _ in range(LAYERS)
    ]
    loss, parts_out = distiller.kv_loss(student_kv=cache, teacher_kv=cache)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert parts_out

    shifted = [{"k": entry["k"] * -1.0, "v": entry["v"]} for entry in cache]
    loss_bad, _ = distiller.kv_loss(student_kv=shifted, teacher_kv=cache)
    assert float(loss_bad) > float(loss)


def test_interface_loss_honours_the_keep_mask():
    parts = make_parts(generative=True)
    distiller = InterfaceDistiller(
        config=InterfaceDistillConfig.from_dict({"enabled": True, "teacher_steps": 2}),
        dream_expert=parts["dream_expert"],
        num_layers=LAYERS,
    )
    dream_seq_len = parts["dream_seq_len"]
    teacher = [
        {"k": torch.randn(2, dream_seq_len, INNER), "v": torch.randn(2, dream_seq_len, INNER)}
        for _ in range(LAYERS)
    ]
    student = [{"k": entry["k"].clone(), "v": entry["v"].clone()} for entry in teacher]
    for entry in student:  # corrupt only the tokens we will mask out
        entry["k"][:, dream_seq_len // 2 :] *= -1.0
        entry["v"][:, dream_seq_len // 2 :] *= -1.0
    keep = torch.zeros(2, dream_seq_len, dtype=torch.bool)
    keep[:, : dream_seq_len // 2] = True
    loss, _ = distiller.kv_loss(student_kv=student, teacher_kv=teacher, keep_mask=keep)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_ema_update_moves_teacher_towards_student():
    parts = make_parts(generative=True)
    distiller = InterfaceDistiller(
        config=InterfaceDistillConfig.from_dict({"enabled": True, "ema_decay": 0.5}),
        dream_expert=parts["dream_expert"],
        num_layers=LAYERS,
    )
    student = parts["dream_expert"]
    with torch.no_grad():
        for param in student.parameters():
            param.add_(1.0)
    before = {name: p.clone() for name, p in distiller.teacher_dream.named_parameters()}
    distiller.update_ema(student)
    moved = [
        not torch.equal(before[name], p)
        for name, p in distiller.teacher_dream.named_parameters()
    ]
    assert any(moved)


def test_generative_expert_without_inputs_matches_the_regression_expert():
    torch.manual_seed(3)
    plain = make_dream_expert(generative=False)
    torch.manual_seed(3)
    promoted = make_dream_expert(generative=True)

    kwargs = dict(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    with torch.no_grad():
        expected = plain.pre_dit(**kwargs)
        actual = promoted.pre_dit(**kwargs)
    assert torch.equal(expected["tokens"], actual["tokens"])
    assert torch.equal(expected["t_mod"], actual["t_mod"])


def test_generative_expert_encodes_targets_into_the_dream_layout():
    expert = make_dream_expert(generative=True)
    batch, offsets = 2, expert.num_future_offsets
    targets = {
        "depth": torch.randn(batch, offsets, 8, 3),
        "dino": torch.randn(batch, offsets, 2, 4, 5),
    }
    latent = expert.encode_targets(targets)
    assert latent.shape == (batch, expert.num_dream_tokens, expert.hidden_dim)
    # Default init_scale is 1: from scratch the branch must actually see its
    # input. an internal run shows what a 0 init costs -- out_scale
    # reached only 0.012 after 416 steps and loss_dream never left the
    # conditional-mean baseline.
    assert torch.count_nonzero(latent) > 0

    # init_scale=0 is still available, for converting a pretrained regression
    # Dream without perturbing it.
    zeroed = make_dream_expert(generative=False)
    GenerativeDreamExpert.promote(
        zeroed, {"enabled": True, "freq_dim": 8, "target_encoder_init_scale": 0.0}
    )
    assert torch.count_nonzero(zeroed.encode_targets(targets)) == 0


def test_generative_pre_dit_consumes_noisy_latent_and_timestep():
    expert = make_dream_expert(generative=True)
    batch = 2
    latent = torch.randn(batch, expert.num_dream_tokens, expert.hidden_dim)
    timestep = torch.full((batch,), 500.0)
    base = expert.pre_dit(batch_size=batch, device=torch.device("cpu"), dtype=torch.float32)
    noisy = expert.pre_dit(
        batch_size=batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
        noisy_latent=latent,
        timestep=timestep,
    )
    torch.testing.assert_close(noisy["tokens"], base["tokens"] + latent)
    # The timestep projection is zero-initialised, so t_mod is untouched at init.
    torch.testing.assert_close(noisy["t_mod"], base["t_mod"])


def test_router_gradients_flow_to_the_gate():
    parts = make_parts()
    router = make_router(parts, mode="learned", rank=4, lambda_budget=1.0, target_keep_ratio=0.2)
    routed = RoutedMoT(mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False, router=router)
    routed.train()
    out = run_mot(routed, parts)
    loss = out["action"].pow(2).mean() + router.budget_loss(routed.last_gates)
    loss.backward()
    grads = [p.grad for p in router.parameters() if p.grad is not None]
    assert grads, "router received no gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def _stub_routed_model(parts):
    """A RoutedWAM shell with only the attributes the Dream rollout touches.

    Building a real one would download Wan2.2-TI2V-5B; the rollout under test
    only needs the Dream expert, the MoT and a scheduler.
    """
    model = object.__new__(RoutedWAM)
    nn.Module.__init__(model)
    model.dream_expert = parts["dream_expert"]
    model.mot = RoutedMoT(
        mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False, router=None
    )
    model.infer_dream_scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000, shift=5.0
    )
    model.train_dream_scheduler = model.infer_dream_scheduler
    return model


def test_dream_rollout_produces_an_interface_and_advances_the_targets():
    parts = make_parts(generative=True)
    model = _stub_routed_model(parts)
    with torch.no_grad():
        for encoder in parts["dream_expert"].target_encoders.values():
            encoder.out_scale.fill_(1.0)

    video_seq_len = parts["video_seq_len"]
    dream_seq_len = parts["dream_seq_len"]
    context_seq_len = video_seq_len + dream_seq_len
    mask = parts["mask"]
    batch = parts["batch"]

    with torch.no_grad():
        video_kv = model.mot.prefill_video_cache(
            video_tokens=parts["embeds"]["video"],
            video_freqs=parts["freqs"]["video"],
            video_t_mod=parts["t_mod"]["video"],
            video_context_payload=None,
            video_attention_mask=mask[:video_seq_len, :video_seq_len],
        )
        initial = model._dream_noise_like_targets(
            batch_size=batch, device=torch.device("cpu"), dtype=torch.float32
        )
        rollout = model._run_dream_rollout(
            num_steps=2,
            scheduler=model.infer_dream_scheduler,
            initial_targets=initial,
            context=None,
            context_mask=None,
            video_kv_cache=video_kv,
            context_attention_mask=mask[:context_seq_len, :context_seq_len],
            video_seq_len=video_seq_len,
            batch_size=batch,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert set(rollout["targets"]) == {"depth", "dino"}
    assert rollout["targets"]["depth"].shape == initial["depth"].shape
    assert rollout["targets"]["dino"].shape == initial["dino"].shape
    # Denoising must actually move the sample.
    assert not torch.allclose(rollout["targets"]["depth"], initial["depth"])

    assert len(rollout["dream_kv"]) == LAYERS
    for entry in rollout["dream_kv"]:
        assert entry["k"].shape == (batch, dream_seq_len, INNER)
        assert entry["v"].shape == (batch, dream_seq_len, INNER)


def test_generative_dream_loss_masks_invalid_futures():
    parts = make_parts(generative=True)
    model = _stub_routed_model(parts)
    model.loss_lambda_dyn = 1.0
    model.loss_lambda_depth = 1.0
    model.loss_lambda_dino = 1.0
    model.loss_lambda_sam = 1.0

    batch, offsets = parts["batch"], parts["dream_expert"].num_future_offsets
    target = {
        "depth": torch.zeros(batch, offsets, 8, 3),
        "dino": torch.zeros(batch, offsets, 2, 4, 5),
    }
    prediction = {name: torch.ones_like(value) for name, value in target.items()}
    # Everything valid: squared error of 1 per element, one modality lambda each.
    loss, _ = model._generative_dream_loss(
        prediction, target, future_valid_mask=None, modality_valid_masks=None
    )
    assert float(loss) == pytest.approx(2.0, abs=1e-5)

    # Mark every future invalid: the masked mean must fall back to zero.
    invalid = torch.zeros(batch, offsets)
    loss_masked, _ = model._generative_dream_loss(
        prediction, target, future_valid_mask=invalid, modality_valid_masks=None
    )
    assert float(loss_masked) == pytest.approx(0.0, abs=1e-6)


def test_camera_grouping_matches_the_decoder_two_view_layout():
    """primary/wrist grouping must agree with DenseDreamDecoder.forward_two_view.

    Within one (modality, offset) block the first `camera_token_split[0]` tokens
    are decoded as the primary view and the rest as the wrist view. If the group
    map disagreed, every stage-dependent-perception figure would attribute
    wrist evidence to the primary camera and vice versa -- a wrong conclusion
    that no amount of downstream plotting would reveal.
    """
    ids, names = build_group_ids(
        modalities=["depth", "dino"],
        num_future_offsets=2,
        tokens_per_modality={"depth": 4, "dino": 4},
        granularity="modality_horizon_camera",
        camera_token_split=(2, 2),
    )
    assert names == [
        "depth@t0/primary", "depth@t0/wrist",
        "depth@t1/primary", "depth@t1/wrist",
        "dino@t0/primary", "dino@t0/wrist",
        "dino@t1/primary", "dino@t1/wrist",
    ]
    # depth: [t0 primary x2, t0 wrist x2, t1 primary x2, t1 wrist x2], then dino.
    assert ids.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]

    ids, names = build_group_ids(
        modalities=["depth", "dino"],
        num_future_offsets=2,
        tokens_per_modality={"depth": 4, "dino": 4},
        granularity="modality_camera",
        camera_token_split=(2, 2),
    )
    assert names == ["depth/primary", "depth/wrist", "dino/primary", "dino/wrist"]
    # Horizons collapse, cameras do not.
    assert ids.tolist() == [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]


def test_camera_grouping_requires_a_camera_split():
    with pytest.raises(ValueError, match="camera_token_split"):
        build_group_ids(
            modalities=["depth"],
            num_future_offsets=1,
            tokens_per_modality={"depth": 4},
            granularity="modality_camera",
            camera_token_split=None,
        )
    with pytest.raises(ValueError, match="must sum to"):
        build_group_ids(
            modalities=["depth"],
            num_future_offsets=1,
            tokens_per_modality={"depth": 4},
            granularity="modality_camera",
            camera_token_split=(3, 3),
        )


def _run_split_path(routed, parts):
    """Video-once prefill -> Dream -> cached Action, the split training path."""
    video_seq_len = parts["video_seq_len"]
    dream_seq_len = parts["dream_seq_len"]
    context_seq_len = video_seq_len + dream_seq_len
    mask = parts["mask"]
    video_kv = routed.prefill_video_cache(
        video_tokens=parts["embeds"]["video"],
        video_freqs=parts["freqs"]["video"],
        video_t_mod=parts["t_mod"]["video"],
        video_context_payload=None,
        video_attention_mask=mask[:video_seq_len, :video_seq_len],
    )
    dream_out = routed.forward_dream_with_video_cache(
        dream_tokens=parts["embeds"]["dream"],
        dream_freqs=parts["freqs"]["dream"],
        dream_t_mod=parts["t_mod"]["dream"],
        dream_context_payload=None,
        video_kv_cache=video_kv,
        context_attention_mask=mask[:context_seq_len, :context_seq_len],
        video_seq_len=video_seq_len,
    )
    return routed.forward_action_with_context_cache(
        action_tokens=parts["embeds"]["action"],
        action_freqs=parts["freqs"]["action"],
        action_t_mod=parts["t_mod"]["action"],
        action_context_payload=None,
        context_kv_cache=routed.merge_context_cache(video_kv, dream_out["dream_kv"]),
        attention_mask=mask,
        video_seq_len=video_seq_len,
        dream_seq_len=dream_seq_len,
    )


def test_cached_path_collects_gates_so_the_budget_loss_is_live():
    """Regression: the split path must feed `budget_loss`, not silently skip it.

    `routed_wam` trains through the cached Action path, not `forward`. When the
    gate was only collected in `forward`, `budget_loss` received an empty list,
    returned a constant zero, and the router never felt any pruning pressure --
    visible in training logs as `loss_router_budget=0.0000` with a keep ratio
    frozen at `sigmoid(bias_init)`.
    """
    parts = make_parts()
    router = make_router(
        parts, mode="learned", rank=4, lambda_budget=1.0, target_keep_ratio=0.2
    )
    routed = RoutedMoT(mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False, router=router)
    routed.train()

    _run_split_path(routed, parts)

    assert len(routed.last_gates) == LAYERS, (
        f"expected one gate per layer, got {len(routed.last_gates)}"
    )
    budget = router.budget_loss(routed.last_gates)
    assert budget.requires_grad, "budget loss is detached from the router"
    assert float(budget.detach()) > 0.0, "budget loss is zero: the router gets no pruning pressure"

    budget.backward()
    grads = [p.grad for p in router.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads), (
        "budget loss produced no router gradient"
    )


def test_cached_path_does_not_leak_gates_between_forwards():
    parts = make_parts()
    router = make_router(parts, mode="learned", rank=4, lambda_budget=1.0)
    routed = RoutedMoT(mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False, router=router)
    routed.eval()
    with torch.no_grad():
        _run_split_path(routed, parts)
        first = len(routed.last_gates)
        _run_split_path(routed, parts)
    assert len(routed.last_gates) == first == LAYERS


def test_task_configs_compose_and_select_the_routed_factory():
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    expected = {
        "routed_wam_libero_goal": False,
        "routed_wam_libero_goal_distill": True,
        "routed_wam_libero_4suite": False,
    }
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        for task, distill_enabled in expected.items():
            cfg = compose(config_name="train", overrides=[f"task={task}"])
            model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
            assert model_cfg["_target_"] == "fastwam.routed_runtime.create_routed_wam"
            # Regression Dream by default: flow matching in target space is not
            # well posed through the 57x Dream bottleneck (measured, E1-E6).
            assert model_cfg["generative_dream"]["enabled"] is False
            assert float(model_cfg["generative_dream"]["target_encoder_init_scale"]) == 1.0
            assert model_cfg["online_dream_targets"]["enabled"] is True
            assert model_cfg["interface_distill"]["enabled"] is distill_enabled
            # The split path cannot denoise future video frames.
            assert float(model_cfg["loss"]["lambda_video"]) == 0.0
            assert bool(cfg.freeze_video_expert) is True


def test_interface_convergence_measures_both_drifts():
    """The measurement core must run against the real Dream rollout.

    Only the shapes and the monotone end-point are asserted here -- whether the
    interface actually settles before the output is the empirical question the
    script exists to answer, and a randomly initialised model cannot answer it.
    """
    from experiments.analysis.interface_convergence import measure_convergence

    parts = make_parts(generative=True)
    model = _stub_routed_model(parts)
    with torch.no_grad():
        for encoder in parts["dream_expert"].target_encoders.values():
            encoder.out_scale.fill_(1.0)

    video_seq_len = parts["video_seq_len"]
    context_seq_len = video_seq_len + parts["dream_seq_len"]
    mask = parts["mask"]
    with torch.no_grad():
        video_kv = model.mot.prefill_video_cache(
            video_tokens=parts["embeds"]["video"],
            video_freqs=parts["freqs"]["video"],
            video_t_mod=parts["t_mod"]["video"],
            video_context_payload=None,
            video_attention_mask=mask[:video_seq_len, :video_seq_len],
        )
        result = measure_convergence(
            model,
            num_steps=4,
            video_kv_cache=video_kv,
            context_attention_mask=mask[:context_seq_len, :context_seq_len],
            video_seq_len=video_seq_len,
            context=None,
            context_mask=None,
            batch_size=parts["batch"],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert len(result["per_step"]) == 4
    # Drift is measured against the final step, so the final row must be zero.
    assert result["per_step"][-1]["interface_drift"] == pytest.approx(0.0, abs=1e-6)
    assert result["per_step"][-1]["output_drift"] == pytest.approx(0.0, abs=1e-6)
    assert result["per_step"][0]["output_drift"] > 0.0


def test_every_model_config_key_is_accepted_by_the_factory():
    """Guard against a config key silently falling through to the dense factory.

    `create_routed_wam` forwards `**dream_fastwam_kwargs` to
    `create_dream_fastwam`, so a routed-only key that is not named explicitly
    reaches the wrong function and the job dies at construction time with
    `got an unexpected keyword argument` -- after queueing, after the image
    pulls, after 16 ranks have started (an internal run).

    Nothing here builds a model, so it runs without Wan2.2 weights.
    """
    pytest.importorskip("hydra")
    import inspect

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from fastwam.routed_runtime import create_routed_wam
    from fastwam.runtime import create_dream_fastwam
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=routed_wam_libero_goal"])
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    routed_params = set(inspect.signature(create_routed_wam).parameters)
    dense_params = set(inspect.signature(create_dream_fastwam).parameters)
    accepted = routed_params | dense_params

    unknown = sorted(k for k in model_cfg if k != "_target_" and k not in accepted)
    assert not unknown, (
        f"model config keys {unknown} are accepted by neither create_routed_wam nor "
        "create_dream_fastwam; they would be forwarded to the wrong factory"
    )
    # And the routed-only keys must be named on the routed factory specifically,
    # not swallowed by **kwargs and passed down to the dense one.
    for key in ("router", "interface_distill", "generative_dream",
                "dream_scheduler", "online_dream_targets", "finetune_action_only"):
        assert key in routed_params, f"{key} must be an explicit create_routed_wam parameter"


def test_dream_step_works_without_a_noisy_target():
    """The regression Dream path must run through `_dream_step` too.

    The generative and regression objectives were branched in
    `_split_training_loss`, but `_dream_step` still called `encode_targets`
    unconditionally -- which raises when there is no noisy target. Every rank of
    jobs  and its five siblings died on it, and no test
    caught it because the stub tests only ever exercised the generative path.
    """
    parts = make_parts(generative=False)          # regression Dream
    model = _stub_routed_model(parts)
    video_seq_len = parts["video_seq_len"]
    context_seq_len = video_seq_len + parts["dream_seq_len"]
    mask = parts["mask"]

    with torch.no_grad():
        video_kv = model.mot.prefill_video_cache(
            video_tokens=parts["embeds"]["video"],
            video_freqs=parts["freqs"]["video"],
            video_t_mod=parts["t_mod"]["video"],
            video_context_payload=None,
            video_attention_mask=mask[:video_seq_len, :video_seq_len],
        )
        out = model._dream_step(
            noisy_targets=None,                   # the regression case
            timestep=None,
            context=None,
            context_mask=None,
            video_kv_cache=video_kv,
            context_attention_mask=mask[:context_seq_len, :context_seq_len],
            video_seq_len=video_seq_len,
            batch_size=parts["batch"],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert set(out["prediction"]) == {"depth", "dino"}
    assert out["prediction"]["depth"].shape[:2] == (parts["batch"], parts["dream_expert"].num_future_offsets)
    assert len(out["dream_kv"]) == LAYERS
    for entry in out["dream_kv"]:
        assert entry["k"].shape == (parts["batch"], parts["dream_seq_len"], INNER)


# ---------------------------------------------------------------------------
# TensorBoard mirroring
# ---------------------------------------------------------------------------
def test_routed_trainer_mirrors_scalars_to_tensorboard():
    """Every logged scalar must reach TensorBoard, including router statistics.

    `_wandb_log` is the single funnel the base trainer pushes metrics through, so
    hooking it is what keeps the two sinks in lockstep. The router group tags
    (`router_keep_dino@t1/wrist`) are the ones that matter here -- they are the
    reason the mirror exists, and they contain `@` and `/`, so a tag-sanitising
    bug would drop exactly the series being investigated.
    """
    from fastwam.routed_trainer import RoutedWan22Trainer

    class FakeWriter:
        def __init__(self):
            self.scalars = []
            self.flushed = 0
            self.closed = False

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, float(value), step))

        def flush(self):
            self.flushed += 1

        def close(self):
            self.closed = True

    # Built without __init__: a real trainer needs accelerate, a dataset and a
    # 5B model, none of which this behaviour depends on.
    trainer = object.__new__(RoutedWan22Trainer)
    writer = FakeWriter()
    trainer._tb_writer = writer
    trainer.global_step = 70
    trainer.wandb_run = None

    payload = {
        "train/loss": 0.5,
        "train/router_gate_mean": 0.31,
        "train/router_keep_dino@t1/wrist": 0.678,
        "train/not_a_scalar": object(),
    }
    trainer._wandb_log(payload)

    tags = {tag for tag, _, _ in writer.scalars}
    assert "train/router_keep_dino@t1/wrist" in tags, (
        "router group tags were dropped: these are the curves the mirror exists for"
    )
    assert "train/router_gate_mean" in tags
    assert "train/not_a_scalar" not in tags, "non-scalar payloads must be skipped, not crash"
    assert all(step == 70 for _, _, step in writer.scalars)
    assert writer.flushed >= 1, "events not flushed: a running job would show nothing"

    trainer._finish_wandb()
    assert writer.closed and trainer._tb_writer is None


def test_routed_trainer_without_tensorboard_is_a_noop():
    """A missing writer must not break logging -- a 26 h job cannot die for a chart."""
    from fastwam.routed_trainer import RoutedWan22Trainer

    trainer = object.__new__(RoutedWan22Trainer)
    trainer._tb_writer = None
    trainer.global_step = 1
    trainer.wandb_run = None
    trainer._wandb_log({"train/loss": 1.0})
    trainer._finish_wandb()
