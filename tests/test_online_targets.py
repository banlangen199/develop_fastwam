"""Tests for online dream-target extraction.

The one property that cannot be allowed to drift is the layout: online targets
must be byte-compatible with the token order the Dream decoder reconstructs, or
the loss would be computed against a differently-ordered tensor and nothing
would report it. `test_patchify_matches_unfold` pins that against `F.unfold`,
which is the convention both sides of the pipeline are written to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.models.wan22.routed_wam.online_targets import (  # noqa: E402
    DEPTH_PATCH,
    OnlineDreamTargets,
    OnlineTargetConfig,
    patchify_depth,
)


def test_patchify_matches_unfold():
    """Pin the layout to `F.unfold`, which is the decoder's own convention.

    The Dream decoder consumes depth as ``token_feature`` with 512 tokens of 64
    dims, and the two-view path splits those tokens into a left and a right
    16x16 camera grid. `F.unfold` with stride == kernel is exactly that
    row-major patch order, so asserting against it pins the online extractor to
    the layout the decoder reconstructs -- without coupling this test to which
    side of the pipeline happens to do the patchifying.
    """
    image = torch.randn(3, 128, 256)
    mine = patchify_depth(image, DEPTH_PATCH)
    theirs = torch.stack(
        [
            F.unfold(
                image[i].unsqueeze(0).unsqueeze(0),
                kernel_size=DEPTH_PATCH,
                stride=DEPTH_PATCH,
            )
            .squeeze(0)
            .transpose(0, 1)
            .contiguous()
            for i in range(image.shape[0])
        ]
    )
    assert mine.shape == theirs.shape == (3, 512, 64)
    assert torch.equal(mine, theirs)


def test_patchify_rejects_indivisible_shapes():
    with pytest.raises(ValueError, match="divisible"):
        patchify_depth(torch.randn(1, 130, 256), 8)
    with pytest.raises(ValueError, match=r"expected \[B,H,W\]"):
        patchify_depth(torch.randn(1, 3, 128, 256), 8)


def test_split_cameras_follows_the_horizontal_concat_convention():
    frames = torch.arange(2 * 3 * 4 * 8).float().reshape(2, 3, 4, 8)
    primary, wrist = OnlineDreamTargets.split_cameras(frames)
    assert torch.equal(primary, frames[..., :4])
    assert torch.equal(wrist, frames[..., 4:])
    with pytest.raises(ValueError, match="even"):
        OnlineDreamTargets.split_cameras(torch.randn(1, 3, 4, 7))


def test_config_rejects_unsupported_modalities_and_missing_models():
    with pytest.raises(ValueError, match="sam and dyn"):
        OnlineTargetConfig.from_dict({"enabled": True, "modalities": ["sam"]})
    with pytest.raises(ValueError, match="dino_model"):
        OnlineTargetConfig.from_dict({"enabled": True, "modalities": ["dino"]})
    with pytest.raises(ValueError, match="depth_model"):
        OnlineTargetConfig.from_dict(
            {"enabled": True, "modalities": ["depth"], "dino_model": "x"}
        )
    with pytest.raises(ValueError, match="video_range"):
        OnlineTargetConfig.from_dict({"video_range": "bogus"})
    # Disabled configs may omit everything.
    assert OnlineTargetConfig.from_dict(None).enabled is False


class _StubDino(torch.nn.Module):
    """Stands in for DINOv2: returns CLS + 256 patch tokens of width 768."""

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        tokens = torch.arange(batch * 257 * 768, dtype=torch.float32)
        return type("O", (), {"last_hidden_state": tokens.reshape(batch, 257, 768)})()


class _StubDepth(torch.nn.Module):
    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        return type(
            "O", (), {"predicted_depth": torch.ones(batch, 37, 37) * 2.0}
        )()


def _stub_module(modalities=("dino", "depth")):
    config = OnlineTargetConfig(
        enabled=True, modalities=tuple(modalities), dino_model="stub", depth_model="stub"
    )
    module = OnlineDreamTargets.__new__(OnlineDreamTargets)
    torch.nn.Module.__init__(module)
    module.config = config
    module.dino = _StubDino() if "dino" in modalities else None
    module.depth = _StubDepth() if "depth" in modalities else None
    module.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    module.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    return module


def test_forward_produces_the_configured_target_shapes():
    module = _stub_module()
    # 9 video frames, two 224x224 cameras concatenated along width.
    video = torch.rand(2, 3, 9, 224, 448) * 2 - 1
    out = module(
        video, future_offsets=[16, 32], action_video_freq_ratio=4,
        image_is_pad=torch.zeros(2, 33, dtype=torch.bool),
    )
    assert out["dino"].shape == (2, 2, 16, 32, 768)
    assert out["depth"].shape == (2, 2, 512, 64)
    assert out["future_valid_mask"].shape == (2, 2)
    assert bool(out["future_valid_mask"].all())
    assert torch.equal(out["dino_valid_mask"], out["future_valid_mask"])


def test_padded_futures_are_marked_invalid():
    """A future that lands on padding must not supervise against a repeat frame."""
    module = _stub_module()
    video = torch.rand(2, 3, 9, 224, 448) * 2 - 1
    pad = torch.zeros(2, 33, dtype=torch.bool)
    pad[0, 32] = True                      # sample 0's +32 future is padding
    out = module(
        video, future_offsets=[16, 32], action_video_freq_ratio=4, image_is_pad=pad
    )
    assert out["future_valid_mask"].tolist() == [[True, False], [True, True]]


def test_unindexable_pad_mask_raises_instead_of_zeroing_everything():
    """An offset we cannot validate is a config error, not "all futures invalid".

    Silently returning an all-false mask makes the masked mean collapse to 0/1,
    so `loss_dream` sits at exactly 0.0000 while every other metric looks
    healthy -- the failure mode this whole verification stack exists to catch.
    """
    module = _stub_module()
    video = torch.rand(1, 3, 9, 224, 448)
    short_pad = torch.zeros(1, 12, dtype=torch.bool)      # neither 33 nor 9
    with pytest.raises(ValueError, match="Refusing to silently mark"):
        module(video, future_offsets=[16], action_video_freq_ratio=4, image_is_pad=short_pad)


def test_pad_mask_indexed_by_video_frame_is_detected():
    """A [B, 9] mask is video-indexed, so offset 32 maps to column 8."""
    module = _stub_module()
    video = torch.rand(2, 3, 9, 224, 448)
    pad = torch.zeros(2, 9, dtype=torch.bool)
    pad[0, 8] = True                                       # video frame 8 == offset 32
    out = module(video, future_offsets=[16, 32], action_video_freq_ratio=4, image_is_pad=pad)
    assert out["future_valid_mask"].tolist() == [[True, False], [True, True]]


def test_offsets_must_land_on_a_sampled_video_frame():
    module = _stub_module()
    video = torch.rand(1, 3, 9, 224, 448)
    with pytest.raises(ValueError, match="not a multiple of"):
        module(video, future_offsets=[17], action_video_freq_ratio=4)
    with pytest.raises(ValueError, match="only 9 frames"):
        module(video, future_offsets=[40], action_video_freq_ratio=4)


def test_video_range_conversion():
    module = _stub_module()
    tanh = torch.tensor([[-1.0, 0.0, 1.0]])
    assert torch.allclose(module._to_unit(tanh), torch.tensor([[0.0, 0.5, 1.0]]))
    module.config = OnlineTargetConfig(
        enabled=True, video_range="unit", dino_model="s", depth_model="s"
    )
    unit = torch.tensor([[0.0, 0.5, 1.0]])
    assert torch.allclose(module._to_unit(unit), unit)


def test_dino_grid_orders_cameras_primary_then_wrist():
    """The width axis must be [primary | wrist], matching the decoder split."""
    module = _stub_module(modalities=("dino",))
    video = torch.rand(1, 3, 9, 224, 448) * 2 - 1
    out = module(video, future_offsets=[16], action_video_freq_ratio=4)
    grid = out["dino"][0, 0]                       # [16, 32, 768]
    primary_half, wrist_half = grid[:, :16], grid[:, 16:]
    assert primary_half.shape == wrist_half.shape == (16, 16, 768)


def test_depth_normalisation_is_affine_invariant():
    """Depth-Anything output is defined up to an affine map; the target must not be.

    Without this the dream loss spends capacity fitting a per-frame scale the
    model cannot infer from the input.
    """
    depth = torch.randn(4, 16, 16) * 5 + 100
    normed = OnlineDreamTargets.normalise_depth(depth, "robust")
    rescaled = OnlineDreamTargets.normalise_depth(depth * 3.7 - 42.0, "robust")
    assert torch.allclose(normed, rescaled, atol=1e-4)
    # Centred on the median by construction.
    assert torch.allclose(
        normed.flatten(1).median(dim=1).values, torch.zeros(4), atol=1e-6
    )
    # And 'none' really is a no-op.
    assert torch.equal(depth, OnlineDreamTargets.normalise_depth(depth, "none"))


def test_depth_normalisation_survives_an_outlier():
    """Percentiles, not min/max: one speckle must not set the scale."""
    depth = torch.zeros(1, 16, 16)
    depth[0, 0, 0] = 1e4
    normed = OnlineDreamTargets.normalise_depth(depth, "robust")
    assert torch.isfinite(normed).all()
    # The 255 unaffected pixels stay at the median, i.e. zero.
    assert float(normed.flatten()[1:].abs().max()) == pytest.approx(0.0, abs=1e-6)


def test_config_rejects_an_unknown_depth_normalisation():
    with pytest.raises(ValueError, match="depth_normalize"):
        OnlineTargetConfig.from_dict({"depth_normalize": "minmax"})
