import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from Visualize.dream_prediction_visualization import (
    fit_episode_projections,
    render_record_grid,
    split_prediction_views,
)
from fastwam.models.wan22.dream_fastwam import DreamFastWAM
from fastwam.models.wan22.routed_wam.model import RoutedWAM
from Visualize.infer_dream_episode import (
    _apply_dream_inference_override,
    _dream_inference_metadata,
)


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
    _prepare_action_cache = DreamFastWAM._prepare_action_cache
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
        self.prefill_return_dream_calls = []
        self.cached_action_calls = 0

    def eval(self):
        return self

    def _encode_input_image_latents_tensor(self, input_image, tiled):
        return torch.zeros((1, 1, 1, 1, 1))

    def encode_prompt(self, _prompt):
        return torch.zeros((1, 2, 4)), torch.ones((1, 2), dtype=torch.bool)

    def _prefill_video_dream_cache(self, *, return_dream, **_):
        self.prefill_return_dream_calls.append(bool(return_dream))
        dream_predictions = None
        if return_dream:
            dream_predictions = {"depth": torch.ones((1, 2, 4, 8))}
        return {"dream_predictions": dream_predictions}

    def _predict_action_noise_with_cache(self, *, latents_action, **_):
        self.cached_action_calls += 1
        return torch.zeros_like(latents_action)


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
    assert fake.prefill_return_dream_calls == [True]
    assert fake.cached_action_calls == 3
    assert output["future_offsets"] == [16, 32]
    assert output["camera_token_split"] == [9, 9]
    assert output["dream_predictions"]["depth"].shape == (2, 4, 8)


def test_default_infer_action_does_not_run_dream_decoder():
    fake = _FakeDreamInference()
    output = DreamFastWAM.infer_action(
        fake,
        prompt="test",
        input_image=torch.zeros((1, 3, 16, 16)),
        action_horizon=4,
        num_inference_steps=3,
    )
    assert fake.prefill_return_dream_calls == [False]
    assert fake.cached_action_calls == 3
    assert set(output) == {"action"}


@pytest.mark.parametrize("steps", [1, 4])
def test_routed_visualization_returns_final_targets_and_reuses_cache(steps):
    fake = _FakeDreamInference()
    fake.dream_expert.generative_enabled = True
    fake.dream_inference_steps = steps
    fake.infer_dream_scheduler = object()
    fake.mot = SimpleNamespace(merge_context_cache=lambda video, dream: [video, dream])
    fake._video_prefill = lambda **kwargs: {
        "video_seq_len": 1, "dream_seq_len": 1,
        "attention_mask": torch.ones(6, 6, dtype=torch.bool), "video_kv": [],
    }
    fake._dream_noise_like_targets = lambda **kwargs: {"depth": torch.zeros(1, 2, 4, 8)}
    rollout_calls = []

    def rollout(**kwargs):
        rollout_calls.append(kwargs["num_steps"])
        return {
            "dream_kv": [],
            "targets": {"depth": torch.full((1, 2, 4, 8), 7.0)},
            "prediction": {"depth": torch.full((1, 2, 4, 8), -3.0)},
        }

    fake._run_dream_rollout = rollout
    fake._prefill_video_dream_cache = RoutedWAM._prefill_video_dream_cache.__get__(fake)
    output = DreamFastWAM.infer_action(
        fake, prompt="test", input_image=torch.zeros(1, 3, 16, 16),
        action_horizon=4, num_inference_steps=3, return_dream_predictions=True,
    )
    assert rollout_calls == [steps]
    assert fake.cached_action_calls == 3
    # Visualization must receive denoised targets, never the velocity prediction.
    assert torch.all(output["dream_predictions"]["depth"] == 7)
    assert output["dream_predictions"]["depth"].device.type == "cpu"
    assert split_prediction_views("depth", output["dream_predictions"]["depth"][0])["image"].shape == (4, 4)
    cfg = OmegaConf.create({"EVALUATION": {"num_inference_steps": 3}})
    assert _dream_inference_metadata(fake, cfg)["dream_inference_steps"] == steps


def test_visualization_dream_override_after_training_config_restore():
    cfg = OmegaConf.create({
        "EVALUATION": {"dream_inference_steps": 8},
        "model": {"generative_dream": {"enabled": True},
                  "dream_scheduler": {"inference_steps": 1}},
    })
    OmegaConf.set_struct(cfg, True)
    _apply_dream_inference_override(cfg)
    assert cfg.model.dream_scheduler.inference_steps == 8
    cfg.EVALUATION.dream_inference_steps = None
    _apply_dream_inference_override(cfg)
    assert cfg.model.dream_scheduler.inference_steps == 8
    cfg.EVALUATION.dream_inference_steps = 2
    cfg.model.generative_dream.enabled = False
    with pytest.raises(ValueError, match="generative RoutedWAM"):
        _apply_dream_inference_override(cfg)


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_visualization_rejects_invalid_dream_steps(steps):
    cfg = OmegaConf.create({"EVALUATION": {"dream_inference_steps": steps}})
    with pytest.raises(ValueError, match="positive integer"):
        _apply_dream_inference_override(cfg)


def test_split_prediction_views_restores_camera_geometry():
    depth = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
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
            "depth": torch.rand((2, 4, 8), generator=generator),
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
    assert set(projections) == {"dino", "sam", "sam_pca", "depth_range"}

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


def test_flow_dynamic_samples_are_not_logits(monkeypatch):
    import Visualize.dream_prediction_visualization as vis
    captured = []
    def capture(rgb, probability, **kwargs):
        captured.append(probability.copy())
        return rgb
    monkeypatch.setattr(vis, "overlay_dynamic_heatmap", capture)
    values = np.array([[-2., 0.], [1., 3.]], dtype=np.float32)
    kwargs = dict(projections={}, alpha=0.5, sam_clusters=2, draw_contours=False)
    vis._render_prediction("dyn", values, np.zeros((2, 2, 3), dtype=np.uint8),
                           prediction_mode="flow_matching", **kwargs)
    np.testing.assert_array_equal(captured[0], [[0., 0.], [1., 1.]])
    vis._render_prediction("dyn", values, np.zeros((2, 2, 3), dtype=np.uint8), **kwargs)
    assert captured[1][0, 1] == 0.5
    np.testing.assert_array_equal(values, [[-2., 0.], [1., 3.]])


def test_sam_pca_render_preserves_raw_predictions():
    record = _record()
    before = {key: value.clone() for key, value in record["dream_predictions"].items()}
    projections = fit_episode_projections([record], sam_clusters=2, max_projection_samples=64)
    render_record_grid(record, projections=projections, panel_size=24, alpha=1.0,
                       sam_clusters=2, draw_contours=False, sam_render_mode="pca")
    for key, value in before.items():
        torch.testing.assert_close(record["dream_predictions"][key], value)
