from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from experiments.libero.threshold_pruning_stats import (
    ThresholdPruningCollector,
    instrument_threshold_model,
)


def _record(layer: int, kept: float, *, total: int = 4, alpha: float = 0.2):
    return {
        "layer": layer,
        "alpha_current": alpha,
        "k_mean": kept,
        "slices": {"dream": [3, 3 + total]},
    }


def test_collector_reports_episode_and_overall_average_pruning():
    collector = ThresholdPruningCollector(configured_alpha=0.2, expected_layers=2)
    collector.begin_episode(0)
    collector.begin_replan(expected_denoise_steps=2)
    collector.capture_denoise_step([_record(0, 3), _record(1, 1)])
    collector.capture_denoise_step([_record(0, 2), _record(1, 2)])
    collector.finish_replan()
    episode = collector.finish_episode(success=True)

    assert episode["layer_evaluations"] == 4
    assert episode["average_total_dream_tokens"] == pytest.approx(4.0)
    assert episode["average_kept_dream_tokens"] == pytest.approx(2.0)
    assert episode["average_pruned_dream_tokens"] == pytest.approx(2.0)
    assert episode["prune_ratio"] == pytest.approx(0.5)
    assert episode["any_tokens_pruned"] is True

    overall = collector.overall_summary()
    assert overall["threshold_enabled"] is True
    assert overall["threshold_executed"] is True
    assert overall["episodes"] == 1
    assert overall["replans"] == 1
    assert overall["denoise_steps"] == 2
    assert overall["average_pruned_dream_tokens"] == pytest.approx(2.0)

    assert collector.layer_records == [
        {
            "episode": 0,
            "success": True,
            "replan": 0,
            "denoise_step": 0,
            "layer": 0,
            "alpha": 0.2,
            "total_dream_tokens": 4.0,
            "kept_dream_tokens": 3.0,
            "pruned_dream_tokens": 1.0,
            "prune_ratio": 0.25,
            "k_std": None,
            "k_min": None,
            "k_max": None,
            "reuse_inference_step": None,
            "mask_refreshed": None,
        },
        {
            "episode": 0,
            "success": True,
            "replan": 0,
            "denoise_step": 0,
            "layer": 1,
            "alpha": 0.2,
            "total_dream_tokens": 4.0,
            "kept_dream_tokens": 1.0,
            "pruned_dream_tokens": 3.0,
            "prune_ratio": 0.75,
            "k_std": None,
            "k_min": None,
            "k_max": None,
            "reuse_inference_step": None,
            "mask_refreshed": None,
        },
        {
            "episode": 0,
            "success": True,
            "replan": 0,
            "denoise_step": 1,
            "layer": 0,
            "alpha": 0.2,
            "total_dream_tokens": 4.0,
            "kept_dream_tokens": 2.0,
            "pruned_dream_tokens": 2.0,
            "prune_ratio": 0.5,
            "k_std": None,
            "k_min": None,
            "k_max": None,
            "reuse_inference_step": None,
            "mask_refreshed": None,
        },
        {
            "episode": 0,
            "success": True,
            "replan": 0,
            "denoise_step": 1,
            "layer": 1,
            "alpha": 0.2,
            "total_dream_tokens": 4.0,
            "kept_dream_tokens": 2.0,
            "pruned_dream_tokens": 2.0,
            "prune_ratio": 0.5,
            "k_std": None,
            "k_min": None,
            "k_max": None,
            "reuse_inference_step": None,
            "mask_refreshed": None,
        },
    ]


def test_collector_distinguishes_executed_threshold_from_actual_pruning():
    collector = ThresholdPruningCollector(configured_alpha=0.2, expected_layers=1)
    collector.begin_episode(0)
    collector.begin_replan(expected_denoise_steps=1)
    collector.capture_denoise_step([_record(0, 4)])
    collector.finish_replan()
    collector.finish_episode(success=False)

    overall = collector.overall_summary()
    assert overall["threshold_executed"] is True
    assert overall["any_tokens_pruned"] is False
    assert overall["average_pruned_dream_tokens"] == pytest.approx(0.0)


def test_collector_fails_when_a_layer_or_denoising_step_is_missing():
    collector = ThresholdPruningCollector(configured_alpha=0.2, expected_layers=2)
    collector.begin_episode(0)
    collector.begin_replan(expected_denoise_steps=2)
    with pytest.raises(RuntimeError, match="every MoT layer"):
        collector.capture_denoise_step([_record(0, 2)])

    collector.abort_replan()
    collector.begin_replan(expected_denoise_steps=2)
    collector.capture_denoise_step([_record(0, 2), _record(1, 2)])
    with pytest.raises(RuntimeError, match="every Action denoising step"):
        collector.finish_replan()


def test_model_instrumentation_captures_every_inference_step():
    @dataclass(frozen=True)
    class Config:
        enabled: bool = True
        alpha: float = 0.2
        log_statistics: bool = False
        save_detailed_tensors: bool = False

    class Mot:
        threshold_config = Config()
        num_layers = 2
        last_threshold_statistics = []
        _collect_training_statistics = False

        def current_alpha(self):
            return self.threshold_config.alpha

    class Model:
        def __init__(self):
            self.mot = Mot()

        def _predict_action_noise_with_cache(self):
            self.mot.last_threshold_statistics = [_record(0, 3), _record(1, 1)]
            return "noise"

        def infer_action(self, *, num_inference_steps):
            for _ in range(num_inference_steps):
                self._predict_action_noise_with_cache()
            return {"action": "result"}

    model = Model()
    collector = ThresholdPruningCollector(configured_alpha=0.2, expected_layers=2)
    instrument_threshold_model(model, collector)
    collector.begin_episode(0)
    assert model.infer_action(num_inference_steps=3) == {"action": "result"}
    episode = collector.finish_episode(success=True)

    assert model.mot._collect_training_statistics is True
    assert model.mot.threshold_config.log_statistics is True
    assert episode["denoise_steps"] == 3
    assert episode["layer_evaluations"] == 6
    assert episode["average_pruned_dream_tokens"] == pytest.approx(2.0)


def test_collector_optionally_retains_proxy_tensors_and_scalars():
    record = _record(0, 1)
    record.update(
        {
            "proxy_head_dim": 16,
            "full_k_mean": 1.0,
            "proxy_k_mean": 2.0,
            "proxy_k_delta_mean": 1.0,
            "proxy_teacher_recall": 1.0,
            "proxy_teacher_precision": 0.5,
            "proxy_teacher_jaccard": 0.5,
            "proxy_mask_agreement": 0.75,
            "proxy_false_negative_mean": 0.0,
            "proxy_false_positive_mean": 1.0,
            "proxy_score_mae": 0.1,
            "proxy_score_rmse": 0.2,
            "full_dense_dream_mass": 0.4,
            "full_teacher_kept_mass": 0.3,
            "proxy_mask_kept_full_mass": 0.35,
            "proxy_full_mass_retention": 0.875,
            "proxy_channel_indices": torch.arange(16),
            "full_score": torch.tensor([[0.7, 0.2, 0.1, 0.0]]),
            "proxy_score": torch.tensor([[0.6, 0.5, 0.1, 0.0]]),
            "full_keep_mask": torch.tensor([[True, False, False, False]]),
            "proxy_keep_mask": torch.tensor([[True, True, False, False]]),
            "full_dense_dream_mass_per_sample": torch.tensor([0.4]),
            "full_teacher_kept_mass_per_sample": torch.tensor([0.3]),
            "proxy_mask_kept_full_mass_per_sample": torch.tensor([0.35]),
            "proxy_full_mass_retention_per_sample": torch.tensor([0.875]),
        }
    )
    collector = ThresholdPruningCollector(
        configured_alpha=0.2,
        expected_layers=1,
        capture_proxy_tensors=True,
    )
    collector.begin_episode(4)
    collector.begin_replan(expected_denoise_steps=1)
    collector.capture_denoise_step([record])
    collector.finish_replan()
    collector.finish_episode(success=True)

    assert collector.layer_records[0]["proxy_teacher_recall"] == 1.0
    assert collector.layer_records[0]["proxy_full_mass_retention"] == 0.875
    assert len(collector.proxy_tensor_records) == 1
    assert collector.proxy_tensor_records[0]["episode"] == 4
    assert collector.proxy_tensor_records[0]["success"] is True
    torch.testing.assert_close(
        collector.proxy_tensor_records[0]["proxy_score"],
        record["proxy_score"],
    )
