from __future__ import annotations

import torch

from fastwam.models.wan22.dream_fastwam.model import DreamFastWAM


def test_silog_matches_dreamvla_formula_per_camera() -> None:
    target = torch.ones(1, 1, 2, 4)
    pred = torch.tensor(
        [[[[0.5, 1.0, 2.0, 1.0], [1.0, 2.0, 1.0, 0.5]]]],
        dtype=torch.float32,
    )

    actual = DreamFastWAM._silog_depth_loss_each(pred, target)

    view_losses = []
    for pred_view in pred.chunk(2, dim=-1):
        diff = torch.log(target[..., :2] + 1e-6) - torch.log(pred_view + 1e-6)
        view_losses.append(torch.sqrt(diff.square().mean() - 0.5 * diff.mean().square()))
    expected = torch.stack(view_losses).mean().reshape(1, 1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_silog_reduces_each_batch_horizon_and_camera_independently() -> None:
    target = torch.ones(2, 2, 2, 4)
    pred = target.clone()
    pred[0, 1, :, :2] = 2.0
    pred[1, 0, :, 2:] = 0.5

    loss = DreamFastWAM._silog_depth_loss_each(pred, target)

    assert loss.shape == (2, 2)
    torch.testing.assert_close(loss[0, 0], torch.zeros_like(loss[0, 0]))
    torch.testing.assert_close(loss[1, 1], torch.zeros_like(loss[1, 1]))
    assert loss[0, 1] > 0
    assert loss[1, 0] > 0


def test_silog_exact_match_has_finite_zero_gradient() -> None:
    pred = torch.ones(2, 2, 3, 6, requires_grad=True)
    target = torch.ones_like(pred)

    loss = DreamFastWAM._silog_depth_loss_each(pred, target).mean()
    loss.backward()

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
