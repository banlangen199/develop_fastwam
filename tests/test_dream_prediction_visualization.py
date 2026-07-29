import inspect
from types import SimpleNamespace

import numpy as np
import torch

from Visualize.dream_prediction_visualization import (
    fit_episode_projections,
    render_record_grid,
    split_prediction_views,
)
from fastwam.models.wan22.dream_fastwam import DreamFastWAM


def test_infer_action_dream_predictions_are_opt_in():
    parameter = inspect.signature(DreamFastWAM.infer_action).parameters[
        "return_dream_predictions"
    ]
    assert parameter.default is False


class _FakeScheduler:
    def build_inference_schedule(self, **_):
        return torch.tensor([1.0, 0.5, 0.1]), torch.tensor([0.1, 0.1, 0.1])

    def step(self, prediction, _delta, latents):
        return latents - prediction


class _FakeDreamInference:
    device = torch.device("cpu")
    torch_dtype = torch.float32
    proprio_dim = None
    use_correlated_noise_infer = False
    action_expert = SimpleNamespace(action_dim=2)
    video_expert = SimpleNamespace(fuse_vae_embedding_in_latents=True)
    dream_expert = SimpleNamespace(
        future_offsets=[16, 32],
        camera_token_split=(9, 9),
    )
    infer_action_scheduler = _FakeScheduler()

    def __init__(self):
        self.return_dream_calls = []

    def eval(self):
        return self

    def _encode_input_image_latents_tensor(self, input_image, tiled):
        return torch.zeros((1, 1, 1, 1, 1))

    def encode_prompt(self, _prompt):
        return torch.zeros((1, 2, 4)), torch.ones((1, 2), dtype=torch.bool)

    def _predict_action_noise(self, *, latents_action, return_dream, **_):
        self.return_dream_calls.append(bool(return_dream))
        action = torch.zeros_like(latents_action)
        if return_dream:
            return action, {"depth": torch.ones((1, 2, 8, 4))}
        return action


def test_infer_action_decodes_dream_only_once():
    fake = _FakeDreamInference()
    output = DreamFastWAM.infer_action(
        fake,
        prompt="test",
        input_image=torch.zeros((1, 3, 16, 16)),
        action_horizon=4,
        num_inference_steps=3,
        return_dream_predictions=True,
    )
    assert fake.return_dream_calls == [True, False, False]
    assert output["future_offsets"] == [16, 32]
    assert output["camera_token_split"] == [9, 9]
    assert output["dream_predictions"]["depth"].shape == (2, 8, 4)


def test_default_infer_action_does_not_run_dream_decoder():
    fake = _FakeDreamInference()
    output = DreamFastWAM.infer_action(
        fake,
        prompt="test",
        input_image=torch.zeros((1, 3, 16, 16)),
        action_horizon=4,
        num_inference_steps=3,
    )
    assert fake.return_dream_calls == [False, False, False]
    assert set(output) == {"action"}


def test_split_prediction_views_restores_camera_geometry():
    depth = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    depth_views = split_prediction_views("depth", depth)
    assert depth_views["image"].shape == (4, 4)
    assert depth_views["wrist_image"].shape == (4, 4)

    dyn = np.arange(8, dtype=np.float32).reshape(8, 1)
    dyn_views = split_prediction_views("dyn", dyn)
    assert dyn_views["image"].shape == (2, 2)
    assert dyn_views["wrist_image"].shape == (2, 2)

    features = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    feature_views = split_prediction_views("dino", features)
    np.testing.assert_array_equal(feature_views["image"], features[:, :2])
    np.testing.assert_array_equal(feature_views["wrist_image"], features[:, 2:])


def _record():
    generator = torch.Generator().manual_seed(0)
    return {
        "metadata": {"replan_index": 0, "env_step": 30, "success": True},
        "rgb": {
            "image": np.full((16, 16, 3), 80, dtype=np.uint8),
            "wrist_image": np.full((16, 16, 3), 120, dtype=np.uint8),
        },
        "future_offsets": [16, 32],
        "dream_predictions": {
            "depth": torch.rand((2, 8, 4), generator=generator),
            "dyn": torch.randn((2, 8, 1), generator=generator),
            "dino": torch.randn((2, 2, 4, 6), generator=generator),
            "sam": torch.randn((2, 2, 4, 4), generator=generator),
        },
    }


def test_episode_projection_and_grid_render_cover_all_modalities():
    record = _record()
    projections = fit_episode_projections(
        [record],
        sam_clusters=2,
        max_projection_samples=64,
    )
    assert set(projections) == {"dino", "sam", "depth_range"}

    frame = render_record_grid(
        record,
        projections=projections,
        panel_size=24,
        alpha=0.5,
        sam_clusters=2,
        draw_contours=False,
    )
    expected_rows = 1 + 4 * 2
    assert frame.dtype == np.uint8
    assert frame.shape == (
        28 + expected_rows * 24 + (expected_rows + 1) * 2,
        2 * 24 + 3 * 2,
        3,
    )
