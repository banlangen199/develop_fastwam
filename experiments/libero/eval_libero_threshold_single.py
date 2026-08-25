"""Evaluate one LIBERO task while auditing Action-to-Dream token pruning.

This keeps the standard rollout behavior, but verifies that threshold pruning
runs at every Action denoising step and MoT layer. It prints per-episode token
counts and saves both the normal result JSON and dedicated pruning JSON/CSV.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fastwam_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fastwam_threshold_matplotlib")

import hydra
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
for source_root in (project_root, project_root / "LIBERO", Path(__file__).resolve().parent):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from experiments.libero import eval_libero_single as base_eval  # noqa: E402
from experiments.libero.threshold_pruning_stats import (  # noqa: E402
    ThresholdPruningCollector,
    instrument_threshold_model,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


def _print_episode_pruning(summary: dict) -> None:
    print(
        "[threshold-pruning] "
        f"episode={summary['episode']} success={summary['success']} "
        f"replans={summary['replans']} denoise_steps={summary['denoise_steps']} "
        f"layer_evaluations={summary['layer_evaluations']} "
        f"avg_pruned={summary['average_pruned_dream_tokens']:.2f}/"
        f"{summary['average_total_dream_tokens']:.2f} "
        f"({summary['prune_ratio']:.2%}) "
        f"avg_kept={summary['average_kept_dream_tokens']:.2f}"
    )


def _write_episode_csv(path: Path, episodes: list[dict]) -> None:
    fields = [
        "episode",
        "success",
        "replans",
        "denoise_steps",
        "layer_evaluations",
        "average_total_dream_tokens",
        "average_kept_dream_tokens",
        "average_pruned_dream_tokens",
        "prune_ratio",
        "min_pruned_dream_tokens",
        "max_pruned_dream_tokens",
        "any_tokens_pruned",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            writer.writerow({field: episode[field] for field in fields})


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_threshold_single_process(cfg: DictConfig):
    start_time = time.time()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")

    training_config_path = base_eval._apply_training_model_config(cfg)
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError(
            "Threshold pruning audit requires EVALUATION.visualize_future_video=false "
            "so evaluation uses the cached Action-only inference path."
        )
    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError("Threshold pruning audit currently requires EVALUATION.env_num=1.")

    model_device = base_eval._resolve_eval_device(cfg)
    model_dtype = base_eval._mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    base_eval._load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    mot = getattr(model, "mot", None)
    threshold_cfg = getattr(mot, "threshold_config", None)
    if threshold_cfg is None:
        raise TypeError("Checkpoint/config did not create a threshold-capable evaluation model.")
    collector = ThresholdPruningCollector(
        configured_alpha=float(mot.current_alpha()),
        expected_layers=int(mot.num_layers),
    )
    instrument_threshold_model(model, collector)
    print(
        "[threshold-pruning] verified configuration: "
        f"enabled={threshold_cfg.enabled} alpha={mot.current_alpha():.6g} "
        f"layers={mot.num_layers}"
    )

    dataset_stats_path = base_eval._resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    base_eval._maybe_load_action_noise_stats(model, cfg, dataset_stats_path)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_cfg is None
        else int(action_horizon_cfg)
    )
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")
    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h, input_w = (int(value) for value in video_size)

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(local_log_dir / "resolved_eval_config.yaml"))
    if training_config_path is not None:
        (local_log_dir / "training_config_path.txt").write_text(
            str(training_config_path) + "\n",
            encoding="utf-8",
        )
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"

    task_suite = base_eval.benchmark.get_benchmark_dict()[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)
    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(
            initial_states[: int(cfg.EVALUATION.num_trials) - len(initial_states)]
        )

    # Keep the standard task/episode implementation and wrap only its episode
    # boundary so pruning totals are printed immediately after each rollout.
    original_run_single_episode = base_eval.run_single_episode

    def tracked_run_single_episode(*args, **kwargs):
        episode_index = int(kwargs.get("episode_idx", 0))
        collector.begin_episode(episode_index)
        try:
            episode_result = original_run_single_episode(*args, **kwargs)
            episode_summary = collector.finish_episode(success=bool(episode_result[0]))
            _print_episode_pruning(episode_summary)
            return episode_result
        except BaseException:
            collector.abort_episode()
            raise

    base_eval.run_single_episode = tracked_run_single_episode
    try:
        task_results = base_eval.run_single_task(
            task=task,
            initial_states=initial_states,
            model=model,
            processor=processor,
            cfg=cfg,
            video_dir=video_dir,
            predicted_video_dir=predicted_video_dir,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
    finally:
        base_eval.run_single_episode = original_run_single_episode

    pruning = collector.overall_summary()
    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": int(cfg.EVALUATION.task_id),
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": start_timestamp,
        "duration": time.time() - start_time,
        **task_results,
        "threshold_pruning": pruning,
    }

    output_dir = local_log_dir / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}"
    result_file = output_dir / f"{prefix}_results.json"
    pruning_file = output_dir / f"{prefix}_pruning.json"
    pruning_csv = output_dir / f"{prefix}_pruning_episodes.csv"
    result_file.write_text(
        json.dumps(results, indent=4, cls=base_eval.NumpyEncoder),
        encoding="utf-8",
    )
    pruning_file.write_text(json.dumps(pruning, indent=4), encoding="utf-8")
    _write_episode_csv(pruning_csv, pruning["episode_summaries"])

    print(
        "[threshold-pruning] overall "
        f"episodes={pruning['episodes']} replans={pruning['replans']} "
        f"avg_pruned={pruning['average_pruned_dream_tokens']:.2f}/"
        f"{pruning['average_total_dream_tokens']:.2f} "
        f"({pruning['prune_ratio']:.2%}) "
        f"any_tokens_pruned={pruning['any_tokens_pruned']}"
    )
    print(f"[threshold-pruning] JSON: {pruning_file}")
    print(f"[threshold-pruning] CSV: {pruning_csv}")
    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    return results


if __name__ == "__main__":
    eval_threshold_single_process()
