#!/usr/bin/env python
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LIBERO_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "libero"
if str(LIBERO_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(LIBERO_EXPERIMENT_DIR))

from Visualize.dream_prediction_visualization import render_saved_episode  # noqa: E402


def _jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON serialize {type(value)}.")


def _copy_rgb_dict(images: dict[str, Any]) -> dict[str, np.ndarray]:
    copied = {}
    for camera in ("image", "wrist_image"):
        if camera not in images:
            raise KeyError(f"LIBERO observation image dict is missing camera={camera!r}.")
        copied[camera] = np.asarray(images[camera], dtype=np.uint8).copy()
    return copied


def _prepare_model_and_processor(cfg: DictConfig):
    from experiments.libero.eval_libero_single import (
        _apply_training_model_config,
        _load_model_checkpoint,
        _maybe_load_action_noise_stats,
        _mixed_precision_to_model_dtype,
        _resolve_dataset_stats_path,
        _resolve_eval_device,
    )
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    training_config_path = _apply_training_model_config(cfg)
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    _maybe_load_action_noise_stats(model, cfg, dataset_stats_path)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    return model, processor, model_device, training_config_path, dataset_stats_path


def _inference_kwargs(
    cfg: DictConfig,
    *,
    prompt: str,
    image: torch.Tensor,
    proprio: torch.Tensor,
    action_horizon: int,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": int(cfg.EVALUATION.num_inference_steps),
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.sigma_shift)
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
        "return_dream_predictions": True,
    }


def _to_executable_action(
    normalized_action: torch.Tensor,
    processor,
    *,
    binarize_gripper: bool,
) -> np.ndarray:
    from experiments.libero.eval_libero_single import _denormalize_action
    from experiments.libero.libero_utils import invert_gripper_action

    action = _denormalize_action(normalized_action, processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if binarize_gripper:
        action[..., -1] = np.sign(action[..., -1])
    return action


def run_one_episode(cfg: DictConfig) -> dict[str, Any]:
    from experiments.libero.eval_libero_single import _get_max_steps, _obs_to_model_input
    from experiments.libero.libero_utils import (
        LIBERO_ENV_RESOLUTION,
        get_libero_dummy_action,
        get_libero_env,
        save_rollout_video,
    )
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.utils.pytorch_utils import set_global_seed
    from libero.libero import benchmark

    if cfg.ckpt is None:
        raise ValueError("Pass ckpt=/path/to/dream_fastwam_checkpoint.pt.")
    set_global_seed(int(cfg.get("seed", 42)), get_worker_init_fn=False)
    (
        model,
        processor,
        model_device,
        training_config_path,
        dataset_stats_path,
    ) = _prepare_model_and_processor(cfg)
    if not hasattr(model, "dream_expert"):
        raise TypeError(
            "Dream episode visualization requires DreamFastWAM, but the loaded model has no dream_expert."
        )

    output_dir = Path(cfg.EVALUATION.output_dir)
    raw_dir = output_dir / "raw_predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    if training_config_path is not None:
        (output_dir / "training_config_path.txt").write_text(
            str(training_config_path) + "\n", encoding="utf-8"
        )

    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    initial_state_index = int(cfg.EVALUATION.initial_state_index)
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    if not 0 <= initial_state_index < len(initial_states):
        raise IndexError(
            f"initial_state_index={initial_state_index} is outside [0,{len(initial_states) - 1}]."
        )
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    obs = env.reset()
    obs = env.set_init_state(initial_states[initial_state_index])

    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if cfg.EVALUATION.get("action_horizon") is None
        else int(cfg.EVALUATION.action_horizon)
    )
    input_h, input_w = (int(x) for x in cfg.data.train.video_size)
    replan_steps = int(cfg.EVALUATION.replan_steps)
    num_steps_wait = int(cfg.EVALUATION.num_steps_wait)
    configured_max_steps = cfg.EVALUATION.get("max_steps")
    max_steps = (
        _get_max_steps(suite_name)
        if configured_max_steps is None
        else int(configured_max_steps)
    )
    if replan_steps <= 0 or action_horizon <= 0:
        raise ValueError("replan_steps and action_horizon must both be positive.")

    pending_actions: list[list[float]] = []
    rollout_images = []
    record_paths = []
    replan_index = 0
    env_step = 0
    success = False
    start_time = time.time()
    prompt = DEFAULT_PROMPT.format(task=task_description)

    while env_step < max_steps + num_steps_wait:
        if env_step < num_steps_wait:
            obs, _, success, _ = env.step(get_libero_dummy_action())
            env_step += 1
            if success:
                break
            continue

        if not pending_actions:
            image, proprio, rgb_images = _obs_to_model_input(
                obs,
                cfg=cfg,
                processor=processor,
                width=input_w,
                height=input_h,
                device=model_device,
                dtype=model.torch_dtype,
            )
            with torch.no_grad():
                prediction = model.infer_action(
                    **_inference_kwargs(
                        cfg,
                        prompt=prompt,
                        image=image,
                        proprio=proprio,
                        action_horizon=action_horizon,
                    )
                )
            if "dream_predictions" not in prediction:
                raise RuntimeError(
                    "Model inference did not return dream_predictions. "
                    "Use a DreamFastWAM checkpoint and the updated inference implementation."
                )
            executable_action = _to_executable_action(
                prediction["action"],
                processor,
                binarize_gripper=bool(cfg.EVALUATION.binarize_gripper),
            )
            pending_actions = executable_action[:replan_steps].tolist()

            record = {
                "metadata": {
                    "replan_index": replan_index,
                    "env_step": env_step,
                    "task_suite_name": suite_name,
                    "task_id": task_id,
                    "initial_state_index": initial_state_index,
                    "task_description": task_description,
                },
                "rgb": _copy_rgb_dict(rgb_images),
                "proprio": proprio.detach().cpu().float(),
                "normalized_action": prediction["action"].detach().cpu().float(),
                "executed_action": torch.from_numpy(executable_action).float(),
                "dream_predictions": prediction["dream_predictions"],
                "future_offsets": list(prediction["future_offsets"]),
                "camera_token_split": prediction.get("camera_token_split"),
            }
            record_path = raw_dir / f"replan_{replan_index:04d}.pt"
            torch.save(record, record_path)
            record_paths.append(record_path)
            replan_index += 1
            rollout_images.append(rgb_images)

        obs, _, success, _ = env.step(pending_actions.pop(0))
        env_step += 1
        if success:
            break

    if hasattr(env, "close"):
        env.close()

    for record_path in record_paths:
        record = torch.load(record_path, map_location="cpu", weights_only=False)
        record["metadata"]["success"] = bool(success)
        torch.save(record, record_path)

    if bool(cfg.VISUALIZATION.get("save_rollout_video", True)) and rollout_images:
        rollout_dir = output_dir / "rollout"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        save_rollout_video(
            rollout_dir,
            rollout_images,
            f"{suite_name}_task{task_id}_state{initial_state_index}",
            success=bool(success),
            task_description=task_description,
        )

    manifest = {
        "checkpoint": str(cfg.ckpt),
        "training_config_path": (
            None if training_config_path is None else str(training_config_path)
        ),
        "dataset_stats_path": str(dataset_stats_path),
        "task_suite_name": suite_name,
        "task_id": task_id,
        "initial_state_index": initial_state_index,
        "task_description": task_description,
        "success": bool(success),
        "env_steps": env_step,
        "num_replans": replan_index,
        "future_offsets": list(model.dream_expert.future_offsets),
        "modalities": list(model.dream_expert.modalities),
        "camera_token_split": (
            None
            if model.dream_expert.camera_token_split is None
            else list(model.dream_expert.camera_token_split)
        ),
        "duration_seconds": time.time() - start_time,
        "raw_dir": str(raw_dir),
    }
    (output_dir / "episode_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_jsonable),
        encoding="utf-8",
    )

    if bool(cfg.VISUALIZATION.get("render_after_inference", True)):
        render_result = render_saved_episode(
            raw_dir,
            output_dir / "rendered",
            fps=int(cfg.VISUALIZATION.fps),
            panel_size=int(cfg.VISUALIZATION.panel_size),
            alpha=float(cfg.VISUALIZATION.overlay_alpha),
            sam_clusters=int(cfg.VISUALIZATION.sam_clusters),
            draw_contours=bool(cfg.VISUALIZATION.draw_contours),
            max_projection_samples=int(cfg.VISUALIZATION.max_projection_samples),
        )
        manifest["render"] = render_result
        (output_dir / "episode_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=_jsonable),
            encoding="utf-8",
        )
    return manifest


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="visualize_dream_episode.yaml",
)
def main(cfg: DictConfig):
    result = run_one_episode(cfg)
    print(json.dumps(result, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
