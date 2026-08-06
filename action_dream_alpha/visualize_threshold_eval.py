"""Run a LIBERO threshold evaluation and visualize pruned Dream tokens.

The normal model path is unchanged. Attention tensors are requested only for
the selected replans, layers, and denoising steps.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for source_root in (ROOT, ROOT / "LIBERO", ROOT / "experiments" / "libero"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from action_dream_alpha.threshold_visualization import save_threshold_record  # noqa: E402
from experiments.libero import visualize_action_attention as attention_vis  # noqa: E402
from experiments.libero.eval_libero_single import (  # noqa: E402
    _apply_training_model_config,
    _get_max_steps,
    _load_model_checkpoint,
    _maybe_load_action_noise_stats,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
)
from experiments.libero.evaluate_dream_fixed_k import _dream_token_metadata  # noqa: E402
from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    invert_gripper_action,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.models.wan22.action_dream_threshold import (  # noqa: E402
    ActionDreamThresholdConfig,
    ThresholdDreamFastWAM,
)
from libero.libero import benchmark  # noqa: E402


def _parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if raw is None or not str(raw).strip():
        return list(default)
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _build_cfg(args: argparse.Namespace):
    overrides = [
        f"task={args.config_name}",
        f"ckpt={args.checkpoint}",
        f"EVALUATION.task_suite_name={args.task_suite}",
        f"EVALUATION.task_id={args.task_id}",
        f"EVALUATION.num_trials={args.num_episodes}",
    ]
    if args.dataset_stats_path:
        overrides.append(f"EVALUATION.dataset_stats_path={args.dataset_stats_path}")
    with initialize_config_dir(config_dir=str((ROOT / "configs").resolve()), version_base="1.3"):
        return compose(config_name="sim_libero.yaml", overrides=overrides)


def _install_threshold(model, alpha: float):
    payload = {
        "enabled": True,
        "alpha": float(alpha),
        "warmup_ratio": 0.0,
        "head_reduce": "mean",
        "action_query_reduce": "max",
        "min_keep_dream_tokens": 0,
        "detach_selection_score": True,
        "log_statistics": False,
        "save_detailed_tensors": False,
    }
    if not hasattr(model.mot, "threshold_config"):
        model = ThresholdDreamFastWAM.from_dense_model(
            model,
            action_dream_threshold=payload,
            finetune_action_only=False,
        )
    else:
        model.mot.threshold_config = ActionDreamThresholdConfig.from_dict(payload)
        model.mot._alpha_current = float(alpha)
    return model.eval()


def _output_dir(args: argparse.Namespace) -> Path:
    checkpoint = Path(args.checkpoint)
    checkpoint_tag = f"{checkpoint.parents[2].name}_{checkpoint.stem}" if len(checkpoint.parents) >= 3 else checkpoint.stem
    root = Path(args.out_dir or "action_dream_alpha/outputs/threshold_eval_visualization")
    path = root / (
        f"{args.task_suite}_task{args.task_id:02d}_alpha{args.alpha:g}_"
        f"{checkpoint_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a shared Action-to-Dream threshold and visualize its token decisions."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--config-name", default="dream_fastwam_threshold_finetune_libero_goal")
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument(
        "--visualize-replans", type=int, default=3,
        help="Save attention for this many initial policy replans per episode.",
    )
    parser.add_argument("--layers", default="0,5,10,15,20,25,29")
    parser.add_argument(
        "--denoise-steps", default=None,
        help="Comma-separated denoise steps; default is first/middle/last.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--max-replans", type=int, default=None)
    parser.add_argument("--dataset-stats-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-headwise-npz", action="store_true",
        help="Do not save exact per-head Action-to-Dream arrays.",
    )
    args = parser.parse_args()
    if args.alpha < 0:
        raise ValueError("alpha must be non-negative.")

    torch.set_grad_enabled(False)
    cfg = _build_cfg(args)
    if args.device:
        cfg.EVALUATION.device = args.device
    training_config_path = _apply_training_model_config(cfg)
    device = str(cfg.EVALUATION.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = _mixed_precision_to_model_dtype(str(cfg.get("mixed_precision", "bf16")))
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    _load_model_checkpoint(model, str(args.checkpoint))
    model = _install_threshold(model.to(device), args.alpha)

    stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(stats_path))
    _maybe_load_action_noise_stats(model, cfg, stats_path)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    total_layers = int(model.mot.num_layers)
    layers = _parse_int_list(args.layers, [0, total_layers // 2, total_layers - 1])
    if any(layer < 0 or layer >= total_layers for layer in layers):
        raise ValueError(f"Layer selection is outside [0,{total_layers}): {layers}")
    inference_steps = int(args.num_inference_steps or cfg.EVALUATION.num_inference_steps)
    denoise_steps = _parse_int_list(
        args.denoise_steps, sorted({0, inference_steps // 2, inference_steps - 1})
    )
    if any(step < 0 or step >= inference_steps for step in denoise_steps):
        raise ValueError(f"Denoise-step selection is outside [0,{inference_steps}): {denoise_steps}")

    output_dir = _output_dir(args)
    run_info = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "training_config_path": None if training_config_path is None else str(training_config_path),
        "alpha": float(args.alpha),
        "threshold_warmup_ratio": 0.0,
        "task_suite": args.task_suite,
        "task_id": int(args.task_id),
        "num_episodes": int(args.num_episodes),
        "visualize_replans": int(args.visualize_replans),
        "layers": layers,
        "denoise_steps": denoise_steps,
        "num_inference_steps": inference_steps,
        "device": device,
        "dtype": str(dtype),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    OmegaConf.save(config=cfg, f=str(output_dir / "resolved_eval_config.yaml"))
    print(f"[threshold-vis] output={output_dir}")

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(int(args.task_id))
    initial_states = suite.get_task_init_states(int(args.task_id))
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    prompt = DEFAULT_PROMPT.format(task=task_description)
    action_horizon = int(cfg.EVALUATION.get("action_horizon") or (int(cfg.data.train.num_frames) - 1))
    video_h, video_w = (int(x) for x in cfg.data.train.video_size)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 10))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 30))
    max_env_steps = int(_get_max_steps(args.task_suite))
    max_replans = args.max_replans or int(math.ceil(max_env_steps / replan_steps))
    all_summaries: list[dict] = []
    episode_results = []
    token_metadata = None

    for episode_index in range(int(args.num_episodes)):
        episode_dir = output_dir / f"episode_{episode_index:03d}"
        episode_dir.mkdir(parents=True)
        env.reset()
        obs = env.set_init_state(initial_states[episode_index % len(initial_states)])
        done = False
        for _ in range(num_steps_wait):
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done:
                break
        env_steps = 0
        replan_index = 0
        while not done and env_steps < max_env_steps and replan_index < max_replans:
            image, proprio, _ = attention_vis._obs_to_model_input(
                obs,
                cfg=cfg,
                processor=processor,
                width=video_w,
                height=video_h,
                device=device,
                dtype=model.torch_dtype,
            )
            capture = replan_index < int(args.visualize_replans)
            action, records = attention_vis._infer_action_with_attention(
                model=model,
                model_type="dream",
                prompt=prompt,
                input_image=image,
                proprio=proprio,
                action_horizon=action_horizon,
                num_inference_steps=inference_steps,
                sigma_shift=None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.sigma_shift),
                rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")),
                tiled=bool(cfg.EVALUATION.get("tiled", False)),
                layers=layers,
                denoise_steps=denoise_steps if capture else [],
                seed=int(args.seed + episode_index * 100000 + replan_index),
            )
            if capture:
                replan_dir = episode_dir / f"replan_{replan_index:03d}"
                replan_dir.mkdir(parents=True)
                attention_vis._save_image(
                    replan_dir / "observation.png", attention_vis._tensor_image_to_uint8(image)
                )
                for record in records:
                    if token_metadata is None:
                        dream_start, dream_end = record["slices"]["dream"]
                        token_metadata = _dream_token_metadata(model, int(dream_end - dream_start))
                        (output_dir / "dream_token_metadata.json").write_text(
                            json.dumps(token_metadata, indent=2), encoding="utf-8"
                        )
                    prefix = replan_dir / (
                        f"layer_{int(record['layer']):02d}_step_{int(record['denoise_step']):02d}"
                    )
                    summary = save_threshold_record(
                        record=record,
                        token_metadata=token_metadata,
                        output_prefix=prefix,
                        save_headwise=not args.no_headwise_npz,
                    )
                    summary.update(
                        {
                            "episode": episode_index,
                            "replan": replan_index,
                            "task": task_description,
                        }
                    )
                    all_summaries.append(summary)

            action_np = attention_vis._denormalize_action(action, processor)[0]
            action_np[..., -1] = action_np[..., -1] * 2 - 1
            action_np = invert_gripper_action(action_np)
            if bool(cfg.EVALUATION.get("binarize_gripper", False)):
                action_np[..., -1] = np.sign(action_np[..., -1])
            for env_action in action_np[:replan_steps].tolist():
                obs, _, done, _ = env.step(env_action)
                env_steps += 1
                if done or env_steps >= max_env_steps:
                    break
            replan_index += 1
        episode_results.append(
            {
                "episode": episode_index,
                "success": bool(done),
                "env_steps": env_steps,
                "replans": replan_index,
            }
        )
        print(
            f"[threshold-vis] episode={episode_index} success={bool(done)} "
            f"steps={env_steps} replans={replan_index}"
        )

    _write_csv(output_dir / "threshold_attention_summary.csv", all_summaries)
    results = {
        **run_info,
        "task_description": task_description,
        "successes": sum(int(item["success"]) for item in episode_results),
        "episodes": episode_results,
        "num_visualized_records": len(all_summaries),
    }
    (output_dir / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        f"[threshold-vis] completed successes={results['successes']}/{args.num_episodes} "
        f"records={len(all_summaries)}"
    )


if __name__ == "__main__":
    main()

