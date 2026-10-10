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


def make_dream_expert(generative: bool = False, image_depth: bool = False) -> DreamQueryExpert:
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
            depth=(dict(target_layout="image", target_shape=[128, 256], patch_size=4)
                   if image_depth else dict(target_layout="token_feature", target_shape=[8, 3])),
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
        future_offsets=list(dream_expert.future_offsets),
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


@pytest.mark.parametrize("gate_type", ["bilinear", "va", "da", "static"])
def test_gate_of_one_reproduces_dense_attention(gate_type):
    parts = make_parts()
    dense = MoT(mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False)
    router = make_router(parts, mode="learned", debug_force_gate=1.0, gate_type=gate_type)
    routed = RoutedMoT(mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False, router=router)
    dense.eval()
    routed.eval()
    with torch.no_grad():
        expected = run_mot(dense, parts)
        actual = run_mot(routed, parts)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("gate_type", ["bilinear", "va", "da", "static"])
def test_gate_of_zero_removes_all_dream_evidence(gate_type):
    parts = make_parts()
    router = make_router(parts, mode="learned", debug_force_gate=0.0, gate_type=gate_type)
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
    # 9e37dbb starts a promoted regression expert with pure-query tokens.
    assert torch.count_nonzero(latent) == 0
    with torch.no_grad():
        for encoder in expert.target_encoders.values():
            encoder.out_scale.fill_(1.0)
    assert torch.count_nonzero(expert.encode_targets(targets)) > 0


def test_generative_image_depth_patch_layout_and_gradient():
    expert = make_dream_expert(generative=True, image_depth=True)
    encoder = expert.target_encoders["depth"]
    decoder = expert.decoders["depth"]
    image = torch.randn(2, 128, 256, requires_grad=True)
    patches = encoder._to_token_layout(image)
    assert patches.shape == (2, 2048, 16)
    torch.testing.assert_close(decoder._unpatchify_image(patches), image)
    primary_idx, wrist_idx = decoder._two_view_query_indices(device=image.device)
    two_view = torch.cat([torch.ones(1, 128, 128), -torch.ones(1, 128, 128)], dim=2)
    view_patches = encoder._to_token_layout(two_view)
    assert torch.all(view_patches[:, primary_idx] == 1)
    assert torch.all(view_patches[:, wrist_idx] == -1)
    with torch.no_grad():
        encoder.out_scale.fill_(1.0)
    targets = {"depth": image.reshape(1, 2, 128, 256), "dino": torch.randn(1, 2, 2, 4, 5)}
    latent = expert.encode_targets(targets)
    assert latent.shape == (1, expert.num_dream_tokens, expert.hidden_dim)
    latent.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert torch.count_nonzero(image.grad) > 0
    with pytest.raises(ValueError, match="image target"):
        encoder(torch.zeros(1, 128, 128))


def test_regression_depth_keeps_nonnegative_output():
    expert = make_dream_expert(image_depth=True)
    decoder = expert.decoders["depth"]
    patches = -torch.ones(1, decoder.num_output_tokens, decoder.feature_dim)
    assert torch.count_nonzero(decoder._unpatchify_image(patches)) == 0
    GenerativeDreamExpert.promote(expert, {"enabled": False})
    assert torch.count_nonzero(decoder._unpatchify_image(patches)) == 0


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
    router = make_router(parts, mode="learned", rank=4)
    routed = RoutedMoT(mixtures=parts["mixtures"], mot_checkpoint_mixed_attn=False, router=router)
    routed.train()
    out = run_mot(routed, parts)
    loss = out["action"].pow(2).mean()
    loss.backward()
    grads = [p.grad for p in router.parameters() if p.grad is not None]
    assert grads, "router received no gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def _cached_action_inputs(routed, parts):
    video_len = parts["video_seq_len"]
    context_len = video_len + parts["dream_seq_len"]
    with torch.no_grad():
        video = routed.prefill_video_cache(
            video_tokens=parts["embeds"]["video"], video_freqs=parts["freqs"]["video"],
            video_t_mod=parts["t_mod"]["video"], video_context_payload=None,
            video_attention_mask=parts["mask"][:video_len, :video_len],
        )
        dream = routed.forward_dream_with_video_cache(
            dream_tokens=parts["embeds"]["dream"], dream_freqs=parts["freqs"]["dream"],
            dream_t_mod=parts["t_mod"]["dream"], dream_context_payload=None,
            video_kv_cache=video,
            context_attention_mask=parts["mask"][:context_len, :context_len],
            video_seq_len=video_len,
        )
    return dict(
        action_tokens=parts["embeds"]["action"], action_freqs=parts["freqs"]["action"],
        action_t_mod=parts["t_mod"]["action"], action_context_payload=None,
        context_kv_cache=routed.merge_context_cache(video, dream["dream_kv"]),
        attention_mask=parts["mask"], video_seq_len=video_len,
        dream_seq_len=parts["dream_seq_len"],
    )


@pytest.mark.parametrize("checkpointed", [False, True])
def test_cached_router_action_loss_trains_every_layer_each_forward(checkpointed):
    parts = make_parts(generative=True)
    model = _stub_routed_model(parts)
    router = make_router(parts, mode="learned", rank=4, warmup_ratio=0.0,
                         group_granularity="modality_horizon_view", gate_type="va")
    model.mot.router = router
    model.mot.mot_checkpoint_mixed_attn = checkpointed
    parts["mixtures"]["action"].use_gradient_checkpointing = checkpointed
    model.eval()
    model.mot.train()  # Matches the trainer: outer model eval, experts train.
    kwargs = _cached_action_inputs(model.mot, parts)
    optimizer = torch.optim.SGD(router.parameters(), lr=1.0)
    for _ in range(2):
        optimizer.zero_grad()
        output = model.mot.forward_action_with_context_cache(**kwargs)
        gates = model.mot.last_group_gates
        assert len(gates) == LAYERS
        assert all(g.requires_grad for g in gates)
        action_loss = output.square().mean()
        loss, metrics = model._add_router_terms(action_loss, {})
        assert loss is action_loss  # Statistics add no regularization loss.
        assert "loss_router_budget" not in metrics
        assert metrics["router_layers"] == LAYERS
        assert metrics["router_strength"] == 1.0
        loss.backward()
        for name, parameter in router.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        optimizer.step()
        assert len(model.mot.last_gates) == LAYERS
        assert len(model.mot.last_group_gates) == LAYERS
        assert len(router.last_statistics) == LAYERS


@pytest.mark.parametrize("gate_type", ["va", "da", "static"])
def test_router_warmup_progress_works_when_outer_model_is_eval(gate_type):
    parts = make_parts(generative=True)
    model = _stub_routed_model(parts)
    router = make_router(parts, mode="learned", warmup_ratio=0.1, gate_type=gate_type)
    model.mot.router = router
    model.router_config = router.config
    model.distiller = None
    model.eval()
    model.mot.train()
    kwargs = _cached_action_inputs(model.mot, parts)
    for step, strength in [(0, 0.0), (5, 0.5), (10, 1.0)]:
        model.set_training_progress_provider(lambda: (step, 100))
        model._refresh_progress()
        assert router.current_strength() == strength
        model.mot.forward_action_with_context_cache(**kwargs)
        loss, _ = model._add_router_terms(torch.zeros(()), {})
        assert loss.item() == 0
        if step == 0:
            for gate in model.mot.last_gates:
                torch.testing.assert_close(gate, torch.ones_like(gate))


def test_enabled_learned_router_rejects_missing_group_records():
    parts = make_parts(generative=True)
    model = _stub_routed_model(parts)
    model.mot.router = make_router(parts, mode="learned")
    with pytest.raises(RuntimeError, match="collected 0 of"):
        model._add_router_terms(torch.zeros(()), {})


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


def test_dream_time_weights_apply_to_each_sample_before_reduction():
    model = _stub_routed_model(make_parts(generative=True))
    for name in ("dyn", "depth", "dino", "sam"):
        setattr(model, f"loss_lambda_{name}", 1.0)
    # Errors [1, 9] with weights [0, 2] should give 9, not
    # mean([1, 9]) * mean([0, 2]) == 5.
    pred = torch.tensor([1.0, 3.0]).reshape(2, 1, 1).expand(2, 2, 1).clone()
    pred.requires_grad_()
    loss, parts = model._generative_dream_loss(
        {"depth": pred}, {"depth": torch.zeros_like(pred)},
        future_valid_mask=None, modality_valid_masks=None,
        sample_weights=torch.tensor([0.0, 2.0]),
    )
    assert loss.item() == pytest.approx(9.0)
    assert parts["loss_depth"].item() == pytest.approx(9.0)
    loss.backward()
    assert torch.count_nonzero(pred.grad[0]) == 0
    assert torch.count_nonzero(pred.grad[1]) == 2

    mask = torch.tensor([[True, True], [True, False]])
    loss, _ = model._generative_dream_loss(
        {"depth": pred}, {"depth": torch.zeros_like(pred)},
        future_valid_mask=None, modality_valid_masks={"depth": mask},
        sample_weights=torch.tensor([0.0, 2.0]),
    )
    assert loss.item() == pytest.approx(6.0)
    # The scheduler returns a scalar for batch size one.
    loss, _ = model._generative_dream_loss(
        {"depth": pred[:1]}, {"depth": torch.zeros_like(pred[:1])},
        future_valid_mask=None, modality_valid_masks=None,
        sample_weights=torch.tensor(2.0),
    )
    assert loss.item() == pytest.approx(2.0)


def test_teacher_rollout_uses_teacher_encoder_time_embedding_and_decoder():
    from unittest.mock import patch

    parts = make_parts(generative=True)
    model = _stub_routed_model(parts)
    distiller = InterfaceDistiller(
        config=InterfaceDistillConfig(enabled=True),
        dream_expert=model.dream_expert, num_layers=LAYERS,
    )
    video_len = parts["video_seq_len"]
    context_len = video_len + parts["dream_seq_len"]
    with torch.no_grad():
        video_kv = model.mot.prefill_video_cache(
            video_tokens=parts["embeds"]["video"],
            video_freqs=parts["freqs"]["video"],
            video_t_mod=parts["t_mod"]["video"],
            video_context_payload=None,
            video_attention_mask=parts["mask"][:video_len, :video_len],
        )
        initial = model._dream_noise_like_targets(
            batch_size=parts["batch"], device=torch.device("cpu"), dtype=torch.float32
        )
        # A teacher rollout must never invoke any of the student's outer layers.
        with (
            patch.object(model.dream_expert, "encode_targets", side_effect=AssertionError("student encoder")),
            patch.object(model.dream_expert, "pre_dit", side_effect=AssertionError("student timestep")),
            patch.object(model.dream_expert, "post_dit", side_effect=AssertionError("student decoder")),
            distiller.use_teacher_dream(model.mot),
        ):
            out = model._run_dream_rollout(
                num_steps=2, scheduler=model.infer_dream_scheduler,
                initial_targets=initial, context=None, context_mask=None,
                video_kv_cache=video_kv,
                context_attention_mask=parts["mask"][:context_len, :context_len],
                video_seq_len=video_len, batch_size=parts["batch"],
                device=torch.device("cpu"), dtype=torch.float32,
            )
        assert torch.isfinite(out["targets"]["depth"]).all()
        assert model.mot.mixtures["dream"] is model.dream_expert


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


def test_pr2_router_gates_individual_dream_keys_within_a_camera_group():
    parts = make_parts()
    router = make_router(parts, mode="learned", rank=4, warmup_ratio=0.0,
                         group_granularity="modality_horizon_camera")
    assert router.config.gate_type == "bilinear"
    assert router.group_mapping[1]["view"] == "wrist"
    assert router.group_mapping[1]["future_offset"] == 0
    assert set(router.state_dict()) == {
        "layer_bias", "group_bias",
        *{f"{projection}.{layer}.weight" for projection in ("query_proj", "key_proj")
          for layer in range(LAYERS)},
    }
    with torch.no_grad():
        router.layer_bias.zero_()
        router.group_bias.zero_()
        router.query_proj[0].weight.fill_(0.25)
        router.key_proj[0].weight.fill_(0.25)
    keys = torch.zeros(2, parts["dream_seq_len"], INNER)
    inputs = dict(layer_idx=0, q_action=torch.ones(2, ACTION_TOKENS, INNER))
    initial = router.gate_for_layer(**inputs, k_dream=keys)["gate"]
    keys[:, 1] = 1.0
    changed = router.gate_for_layer(**inputs, k_dream=keys)["gate"]
    assert router.group_ids[0] == router.group_ids[1]
    assert torch.all(changed[:, 1] > initial[:, 1])
    torch.testing.assert_close(changed[:, 0], initial[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(changed[:, 2:], initial[:, 2:], rtol=0, atol=0)


@pytest.mark.parametrize("cached", [False, True])
def test_pr2_router_checkpointing_preserves_outputs_gradients_and_gate_records(cached):
    results = []
    for checkpointed in (False, True):
        parts = make_parts()
        router = make_router(parts, mode="learned", rank=4, warmup_ratio=0.0,
                             lambda_budget=0.01, group_granularity="modality_horizon_camera")
        parts["mixtures"]["action"].use_gradient_checkpointing = checkpointed
        routed = RoutedMoT(mixtures=parts["mixtures"],
                           mot_checkpoint_mixed_attn=checkpointed, router=router).train()
        if cached:
            output = routed.forward_action_with_context_cache(**_cached_action_inputs(routed, parts))
        else:
            output = run_mot(routed, parts)["action"]
        budget = router.budget_loss(routed.last_gates)
        assert budget.requires_grad and budget.detach() > 0
        (output.square().mean() + budget).backward()
        assert len(routed.last_gates) == len(routed.last_group_gates) == LAYERS
        assert len(router.last_statistics) == LAYERS
        gradients = {}
        for name, parameter in router.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
            gradients[name] = parameter.grad.clone()
        results.append((output.detach(), gradients))
    torch.testing.assert_close(results[0][0], results[1][0])
    for name, gradient in results[0][1].items():
        torch.testing.assert_close(gradient, results[1][1][name])


@pytest.mark.parametrize("checkpointed", [False, True])
def test_pr2_router_trains_with_precomputed_targets_and_regression_dream(checkpointed):
    from fastwam.models.wan22.action_dit import ActionDiT

    torch.manual_seed(11)
    common = dict(hidden_dim=HIDDEN, ffn_dim=FFN, text_dim=HIDDEN, freq_dim=8,
                  eps=1e-6, num_heads=HEADS, attn_head_dim=HEAD_DIM, num_layers=LAYERS)
    video = WanVideoDiT(**common, in_dim=4, out_dim=4, patch_size=(1, 2, 2),
                       has_image_input=False, seperated_timestep=True,
                       video_attention_mask_mode="first_frame_causal")
    action = ActionDiT(**common, action_dim=7, use_gradient_checkpointing=checkpointed)
    dream = DreamQueryExpert(
        **common,
        dream_query=dict(modalities=["depth", "dino"], future_offsets=[16, 32],
                         camera_token_split=[2, 2], n_depth=4, n_dino=4),
        dream_decoder=dict(
            decoder_dim=HIDDEN, decoder_ffn_dim=FFN, num_layers=1, num_heads=HEADS,
            depth=dict(target_layout="image", target_shape=[4, 8], patch_size=2),
            dino=dict(target_layout="grid_feature", target_shape=[2, 4, 5]),
        ),
    )
    mot = MoT(dict(video=video, dream=dream, action=action),
              mot_checkpoint_mixed_attn=checkpointed)
    dense = DreamFastWAM(video_expert=video, action_expert=action, dream_expert=dream,
                         mot=mot, vae=nn.Identity(), text_dim=HIDDEN, proprio_dim=None,
                         loss_lambda_video=0, device="cpu", torch_dtype=torch.float32)
    model = RoutedWAM.from_dense_model(
        dense, generative_dream={"enabled": False}, training_mode="dense_joint",
        router=dict(mode="learned", rank=4, lambda_budget=0.01, warmup_ratio=0.0,
                    group_granularity="modality_horizon_camera"),
    )
    router = model.mot.router
    model.configure_trainable_parameters(freeze_video_expert=True)
    assert not hasattr(model, "online_targets")
    assert model.train_dream_scheduler is None
    # Stub only the data/VAE boundary: cached targets and real tiny experts
    # exercise the complete Dream + Action + router training objective.
    latent = torch.randn(2, 4, 1, 4, 4)
    sample = dict(
        context=torch.randn(2, 3, HIDDEN), context_mask=torch.ones(2, 3, dtype=torch.bool),
        input_latents=latent, first_frame_latents=latent, fuse_vae_embedding_in_latents=True,
        action=torch.randn(2, ACTION_TOKENS, 7),
        action_is_pad=torch.zeros(2, ACTION_TOKENS, dtype=torch.bool),
        dream_targets={"depth": torch.rand(2, 2, 4, 8) + 0.1,
                       "dino": torch.randn(2, 2, 2, 4, 5)},
    )
    model.build_inputs = lambda sample, **_: sample
    loss, metrics = model.training_loss(sample)
    assert metrics["loss_router_budget"] > 0
    assert loss.detach().item() == pytest.approx(
        metrics["loss_action"] + metrics["loss_dream"] + metrics["loss_router_budget"], rel=1e-5
    )
    loss.backward()
    for name, parameter in router.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_task_configs_compose_and_select_the_routed_factory():
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    expected = {
        "routed_wam_libero_10": False,
        "routed_wam_libero_goal": False,
        "routed_wam_libero_goal_distill": True,
        "routed_wam_libero_4suite": False,
        "routed_wam_libero_4suite_distill": True,
        "routed_wam_libero_4suite_full": False,
    }
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        for task, distill_enabled in expected.items():
            cfg = compose(config_name="train", overrides=[f"task={task}"])
            model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
            assert model_cfg["_target_"] == "fastwam.routed_runtime.create_routed_wam"
            assert model_cfg["generative_dream"]["enabled"] is distill_enabled
            assert model_cfg["training_mode"] == ("joint" if distill_enabled else "dense_joint")
            assert "online_dream_targets" not in model_cfg
            assert cfg.data.train.dream_target.enabled is True
            assert model_cfg["dream_query_config"]["dream_decoder"]["depth"]["target_layout"] == "image"
            assert model_cfg["router"]["gate_type"] == "bilinear"
            assert model_cfg["router"]["group_granularity"] == "modality_horizon_camera"
            assert model_cfg["router"]["lambda_budget"] == 0.01
            assert model_cfg["interface_distill"]["enabled"] is distill_enabled
            # The split path cannot denoise future video frames.
            assert float(model_cfg["loss"]["lambda_video"]) == 0.0
            assert bool(cfg.freeze_video_expert) is True


def test_old_routed_weights_can_warm_start_local_decoder_but_other_keys_stay_strict(tmp_path):
    model = _stub_routed_model(make_parts(generative=True))
    model.proprio_encoder = None
    prefix = "mixtures.dream.decoder_conditioners."
    state = {key: value for key, value in model.mot.state_dict().items() if not key.startswith(prefix)}
    checkpoint = tmp_path / "old_routed.pt"
    torch.save({"mot": state}, checkpoint)
    report = model.verify_checkpoint_compatibility(checkpoint)
    assert report["missing_new_parameters"]
    assert all(key.startswith(prefix) for key in report["missing_new_parameters"])
    model.load_checkpoint(checkpoint, strict_shapes=True)
    # New checkpoints include and exactly restore every local decoder weight.
    full_state = {key: value.clone() for key, value in model.mot.state_dict().items()}
    torch.save({"mot": full_state}, checkpoint)
    with torch.no_grad():
        for param in model.dream_expert.decoder_conditioners.parameters():
            param.zero_()
    model.load_checkpoint(checkpoint, strict_shapes=True)
    for key, value in model.mot.state_dict().items():
        torch.testing.assert_close(value, full_state[key])
    del state["mixtures.dream.decoders.depth.output_proj.weight"]
    torch.save({"mot": state}, checkpoint)
    with pytest.raises(RuntimeError, match="missing_pretrained"):
        model.verify_checkpoint_compatibility(checkpoint)


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
                "dream_scheduler", "finetune_action_only"):
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
