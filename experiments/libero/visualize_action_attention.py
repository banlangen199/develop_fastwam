"""Visualize action-query attention for FastWAM and DreamFastWAM.

Examples:

Original FastWAM:
python experiments/libero/visualize_action_attention.py \
  --model_type original \
  --checkpoint runs/libero_uncond_2cam224_1e-4_100m/RUN/checkpoints/weights/step_x.pt \
  --config-name libero_uncond_2cam224_1e-4_100m \
  --task_suite libero_goal \
  --task_id 0 \
  --num_episodes 3 \
  --frames_per_episode 3 \
  --out_dir runs/attention_vis/original_fastwam

DreamFastWAM:
python experiments/libero/visualize_action_attention.py \
  --model_type dream \
  --checkpoint runs/dream_fastwam_libero_goal/RUN/checkpoints/weights/step_x.pt \
  --config-name dream_fastwam_libero_goal \
  --task_suite libero_goal \
  --task_id 0 \
  --num_episodes 3 \
  --frames_per_episode 3 \
  --out_dir runs/attention_vis/dream_fastwam

The script uses a debug-only MoT path that recomputes action-row attention
probabilities. Normal training and evaluation do not return attention.

For each episode it saves per-frame PNGs plus mp4 videos named
attention_video_layer{l}_step{s}.mp4 and source-mass trend plots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fastwam_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fastwam_numba_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.eval_libero_single import (  # noqa: E402
    _apply_training_model_config,
    _denormalize_action,
    _load_model_checkpoint,
    _maybe_load_action_noise_stats,
    _mixed_precision_to_model_dtype,
    _obs_to_model_input,
    _resolve_dataset_stats_path,
)
from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    invert_gripper_action,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.utils.video_io import save_mp4  # noqa: E402
from libero.libero import benchmark  # noqa: E402


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _default_indices(total: int) -> list[int]:
    if total <= 1:
        return [0]
    mid = total // 2
    return sorted(set([0, mid, total - 1]))


def _infer_token_grid(num_tokens: int, image_h: int, image_w: int) -> tuple[int, int]:
    target_ratio = image_w / max(image_h, 1)
    best = None
    for h in range(1, int(num_tokens**0.5) + 1):
        if num_tokens % h != 0:
            continue
        w = num_tokens // h
        score = abs((w / h) - target_ratio)
        if best is None or score < best[0]:
            best = (score, h, w)
    if best is None:
        return 1, num_tokens
    return best[1], best[2]


def _tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().to(device="cpu", dtype=torch.float32)[0]
    image = ((image + 1.0) * 127.5).clamp(0, 255).byte()
    return image.permute(1, 2, 0).numpy()


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.nanmin(x))
    den = float(np.nanmax(x))
    if den <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return x / den


def _save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)
    print(f"Saved {path}")


def _save_overlay(path: Path, rgb: np.ndarray, heatmap: np.ndarray) -> None:
    heatmap = _normalize_map(heatmap)
    hm = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        (rgb.shape[1], rgb.shape[0]), resample=Image.BILINEAR
    )
    color = (plt.get_cmap("jet")(np.asarray(hm) / 255.0)[..., :3] * 255).astype(np.uint8)
    overlay = (0.55 * rgb.astype(np.float32) + 0.45 * color.astype(np.float32)).clip(0, 255).astype(np.uint8)
    _save_image(path, overlay)


def _save_heatmap(path: Path, values: np.ndarray, title: str, xlabel: str = "token index") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 2.8))
    plt.imshow(values[np.newaxis, :], aspect="auto", cmap="viridis")
    plt.yticks([])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.colorbar(fraction=0.035, pad=0.02)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved {path}")


def _save_bar(path: Path, masses: dict[str, float], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(masses.keys())
    vals = [masses[k] for k in names]
    plt.figure(figsize=(5, 3.2))
    plt.bar(names, vals, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(names)])
    plt.ylim(0.0, max(1.0, max(vals) * 1.15 if vals else 1.0))
    plt.ylabel("attention mass")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved {path}")


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), resample=Image.BILINEAR)


def _stitch_panel(paths: list[Path], out_path: Path, title: str) -> None:
    images = [Image.open(path).convert("RGB") for path in paths if path.exists()]
    if not images:
        return
    row_h = 256
    images = [_resize_to_height(image, row_h) for image in images]
    title_h = 34
    canvas_w = sum(image.width for image in images)
    canvas_h = row_h + title_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    try:
        from PIL import ImageDraw

        ImageDraw.Draw(canvas).text((8, 8), title, fill=(0, 0, 0))
    except Exception:
        pass
    x = 0
    for image in images:
        canvas.paste(image, (x, title_h))
        x += image.width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved {out_path}")


def _save_episode_video(panel_paths: list[Path], out_path: Path, fps: int) -> None:
    frames = [Image.open(path).convert("RGB") for path in panel_paths if path.exists()]
    if not frames:
        return
    save_mp4(frames, str(out_path), fps=fps)
    print(f"Saved {out_path}")


def _save_episode_mass_trends(
    *,
    episode_dir: Path,
    stats: list[dict[str, Any]],
    model_type: str,
    representative_layer: int,
    representative_step: int,
) -> None:
    if not stats:
        return
    labels = ["mass_video"]
    if model_type == "dream":
        labels.append("mass_dream")
    labels.append("mass_self")

    rep = [
        item for item in stats
        if int(item["layer_id"]) == representative_layer and int(item["denoise_step"]) == representative_step
    ]
    if rep:
        rep = sorted(rep, key=lambda x: int(x["frame_id"]))
        plt.figure(figsize=(7, 4))
        xs = [int(x["frame_id"]) for x in rep]
        for label in labels:
            plt.plot(xs, [float(x.get(label, 0.0)) for x in rep], marker="o", label=label.replace("mass_", ""))
        plt.xlabel("frame")
        plt.ylabel("attention mass")
        plt.ylim(0.0, 1.0)
        plt.title(f"Source mass over time, layer {representative_layer}, step {representative_step}")
        plt.legend()
        plt.tight_layout()
        path = episode_dir / f"source_mass_over_time_layer{representative_layer}_step{representative_step}.png"
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"Saved {path}")

    for frame_id in sorted({int(x["frame_id"]) for x in stats}):
        frame_stats = [x for x in stats if int(x["frame_id"]) == frame_id]
        plt.figure(figsize=(8, 4))
        xs = [f"L{int(x['layer_id'])}/S{int(x['denoise_step'])}" for x in frame_stats]
        order = np.argsort([int(x["denoise_step"]) * 1000 + int(x["layer_id"]) for x in frame_stats])
        xs = [xs[i] for i in order]
        for label in labels:
            ys = [float(frame_stats[i].get(label, 0.0)) for i in order]
            plt.plot(xs, ys, marker="o", label=label.replace("mass_", ""))
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("attention mass")
        plt.ylim(0.0, 1.0)
        plt.title(f"Source mass by layer/step, frame {frame_id}")
        plt.legend()
        plt.tight_layout()
        path = episode_dir / f"source_mass_by_layer_step_frame{frame_id:03d}.png"
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"Saved {path}")


def _attention_summary(record: dict[str, Any], model_type: str) -> dict[str, Any]:
    probs = record["probs"][0]  # [H, Sa, S]
    slices = record["slices"]
    mean_probs = probs.mean(dim=(0, 1)).numpy()

    video_start, video_end = slices["video"]
    action_start, action_end = slices["action"]
    out = {
        "layer_id": int(record["layer"]),
        "mass_video": float(mean_probs[video_start:video_end].sum()),
        "mass_self": float(mean_probs[action_start:action_end].sum()),
        "num_video_tokens": int(video_end - video_start),
        "num_action_tokens": int(action_end - action_start),
        "video_attention": mean_probs[video_start:video_end],
    }
    if model_type == "dream" and "dream" in slices:
        dream_start, dream_end = slices["dream"]
        out.update(
            {
                "mass_dream": float(mean_probs[dream_start:dream_end].sum()),
                "num_dream_tokens": int(dream_end - dream_start),
                "dream_attention": mean_probs[dream_start:dream_end],
            }
        )
    return out


@torch.no_grad()
def _infer_action_with_attention(
    *,
    model,
    model_type: str,
    prompt: str,
    input_image: torch.Tensor,
    proprio: torch.Tensor | None,
    action_horizon: int,
    num_inference_steps: int,
    sigma_shift: float | None,
    rand_device: str,
    tiled: bool,
    layers: list[int],
    denoise_steps: list[int],
    seed: int | None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    model.eval()
    generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
    latents_action_shape = (1, action_horizon, model.action_expert.action_dim)
    latents_action_base = torch.empty(latents_action_shape, device=rand_device, dtype=torch.float32)
    if getattr(model, "use_correlated_noise_infer", False):
        latents_action = model._sample_action_noise(
            latents_action_base,
            use_correlated_noise=True,
            generator=generator,
        ).to(device=model.device, dtype=model.torch_dtype)
    else:
        latents_action = torch.randn(
            latents_action_shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=model.device, dtype=model.torch_dtype)

    input_image = input_image.to(device=model.device, dtype=model.torch_dtype)
    first_frame_latents = model._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    context, context_mask = model.encode_prompt(prompt)
    if proprio is not None:
        proprio = proprio.to(device=model.device, dtype=model.torch_dtype)
        context, context_mask = model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )

    infer_timesteps_action, infer_deltas_action = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=model.device,
        dtype=latents_action.dtype,
        shift_override=sigma_shift,
    )
    selected_steps = set(int(x) for x in denoise_steps)
    captured: list[dict[str, Any]] = []
    for step_idx, (step_t_action, step_delta_action) in enumerate(zip(infer_timesteps_action, infer_deltas_action)):
        timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=model.device)
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=model.device)
        if step_idx in selected_steps:
            out = model._predict_joint_noise(
                latents_video=first_frame_latents,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=None,
                return_action_attention=True,
                attention_layers=layers,
            )
            pred_action = out[1]
            for record in out[2]:
                item = dict(record)
                item["denoise_step"] = int(step_idx)
                captured.append(item)
        else:
            pred_action = model._predict_action_noise(
                first_frame_latents=first_frame_latents,
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
        latents_action = model.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

    return latents_action[0].detach().to(device="cpu", dtype=torch.float32), captured


def _save_attention_outputs(
    *,
    frame_dir: Path,
    model_type: str,
    episode_id: int,
    frame_id: int,
    model_rgb: np.ndarray,
    records: list[dict[str, Any]],
    concat_multi_camera: str,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], Path]]:
    h, w = model_rgb.shape[:2]
    if concat_multi_camera == "horizontal":
        primary_rgb = model_rgb[:, : w // 2]
        wrist_rgb = model_rgb[:, w // 2 :]
    else:
        primary_rgb = model_rgb
        wrist_rgb = None
    _save_image(frame_dir / "rgb_primary.png", primary_rgb)
    if wrist_rgb is not None:
        _save_image(frame_dir / "rgb_wrist.png", wrist_rgb)

    stats = []
    panel_paths: dict[tuple[int, int], Path] = {}
    for record in records:
        summary = _attention_summary(record, model_type=model_type)
        layer = int(summary["layer_id"])
        step = int(record["denoise_step"])
        video_attn = summary.pop("video_attention")
        grid_h, grid_w = _infer_token_grid(len(video_attn), h, w)
        video_map = video_attn.reshape(grid_h, grid_w)
        if concat_multi_camera == "horizontal" and grid_w >= 2:
            primary_map = video_map[:, : grid_w // 2]
            wrist_map = video_map[:, grid_w // 2 :]
        else:
            primary_map = video_map
            wrist_map = None
        primary_overlay_path = frame_dir / f"attn_overlay_primary_layer{layer}_step{step}.png"
        _save_overlay(primary_overlay_path, primary_rgb, primary_map)
        panel_items = [primary_overlay_path]
        if wrist_rgb is not None and wrist_map is not None:
            wrist_overlay_path = frame_dir / f"attn_overlay_wrist_layer{layer}_step{step}.png"
            _save_overlay(wrist_overlay_path, wrist_rgb, wrist_map)
            panel_items.append(wrist_overlay_path)

        masses = {"video/image": summary["mass_video"]}
        if model_type == "dream":
            masses["dream"] = summary.get("mass_dream", 0.0)
        masses["action-self"] = summary["mass_self"]
        source_mass_path = frame_dir / f"source_mass_layer{layer}_step{step}.png"
        _save_bar(source_mass_path, masses, f"layer {layer}, step {step}")
        panel_items.append(source_mass_path)
        if model_type == "dream" and "dream_attention" in summary:
            dream_attn = summary.pop("dream_attention")
            dream_path = frame_dir / f"dream_attention_layer{layer}_step{step}.png"
            _save_heatmap(
                dream_path,
                dream_attn,
                title=f"Action to dream attention, layer {layer}, step {step}",
                xlabel="dream token index",
            )
            panel_items.append(dream_path)
        summary.update(
            {
                "episode_id": int(episode_id),
                "frame_id": int(frame_id),
                "model_type": model_type,
                "denoise_step": step,
            }
        )
        stats.append(summary)
        panel_path = frame_dir / f"panel_layer{layer}_step{step}.png"
        _stitch_panel(
            panel_items,
            panel_path,
            title=f"episode {episode_id}, frame {frame_id}, layer {layer}, denoise step {step}",
        )
        panel_paths[(layer, step)] = panel_path

    stats_path = frame_dir / "attention_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved {stats_path}")
    return stats, panel_paths


def _build_cfg(args: argparse.Namespace):
    config_dir = str((project_root / "configs").resolve())
    overrides = [
        f"task={args.config_name}",
        f"ckpt={args.checkpoint}",
        f"EVALUATION.task_suite_name={args.task_suite}",
        f"EVALUATION.task_id={args.task_id}",
        "EVALUATION.num_trials=1",
    ]
    if args.dataset_stats_path:
        overrides.append(f"EVALUATION.dataset_stats_path={args.dataset_stats_path}")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(config_name="sim_libero.yaml", overrides=overrides)


def _checkpoint_tag(checkpoint: str) -> str:
    path = Path(checkpoint)
    step = path.stem
    run = path.parents[2].name if len(path.parents) >= 3 else path.parent.name
    return f"{run}_{step}".replace("/", "_")


def _resolve_output_root(args: argparse.Namespace) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    leaf = (
        f"{args.task_suite}_task{int(args.task_id):02d}_"
        f"{_checkpoint_tag(args.checkpoint)}_{timestamp}"
    )
    if args.out_dir is None:
        base = Path("runs/attention_vis") / args.model_type / args.config_name
    else:
        base = Path(args.out_dir)
    out_root = base / leaf
    if out_root.exists() and not args.overwrite:
        suffix = 1
        candidate = Path(f"{out_root}_{suffix:02d}")
        while candidate.exists():
            suffix += 1
            candidate = Path(f"{out_root}_{suffix:02d}")
        out_root = candidate
    out_root.mkdir(parents=True, exist_ok=bool(args.overwrite))
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize FastWAM/DreamFastWAM action attention in LIBERO.")
    parser.add_argument("--model_type", choices=["original", "dream"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", required=True, help="Hydra task config name, e.g. dream_fastwam_libero_goal.")
    parser.add_argument("--task_suite", default="libero_goal")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--frames_per_episode", type=int, default=3)
    parser.add_argument("--layers", default=None, help="Comma-separated layer ids. Default: first, middle, last.")
    parser.add_argument("--denoise_steps", default=None, help="Comma-separated denoise step ids. Default: early/mid/late.")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--dataset_stats_path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video_fps", type=int, default=2)
    parser.add_argument("--no_video", action="store_true", help="Disable episode mp4 generation.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing resolved run dir.")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    cfg = _build_cfg(args)
    if args.device is not None:
        cfg.EVALUATION.device = args.device
    training_config_path = _apply_training_model_config(cfg)
    device = str(cfg.EVALUATION.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    _load_model_checkpoint(model, str(args.checkpoint))
    model = model.to(device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    _maybe_load_action_noise_stats(model, cfg, dataset_stats_path)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    total_layers = int(model.mot.num_layers)
    layers = _parse_int_list(args.layers) or _default_indices(total_layers)
    num_inference_steps = int(args.num_inference_steps or cfg.get("eval_num_inference_steps", 10))
    denoise_steps = _parse_int_list(args.denoise_steps) or _default_indices(num_inference_steps)
    layers = [x for x in layers if 0 <= x < total_layers]
    denoise_steps = [x for x in denoise_steps if 0 <= x < num_inference_steps]
    if not layers or not denoise_steps:
        raise ValueError(f"No valid layers/denoise_steps selected: layers={layers}, steps={denoise_steps}")

    action_horizon = int(cfg.EVALUATION.get("action_horizon") or (int(cfg.data.train.num_frames) - 1))
    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h, input_w = int(video_size[0]), int(video_size[1])
    concat_multi_camera = str(cfg.data.train.get("concat_multi_camera", "horizontal"))
    out_root = _resolve_output_root(args)
    print(f"[attention-vis] output_dir={out_root}")
    OmegaConf.save(config=cfg, f=str(out_root / "resolved_attention_config.yaml"))
    if training_config_path is not None:
        (out_root / "training_config_path.txt").write_text(
            str(training_config_path) + "\n",
            encoding="utf-8",
        )
    run_info = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "config_name": args.config_name,
        "training_config_path": None if training_config_path is None else str(training_config_path),
        "task_suite": args.task_suite,
        "task_id": int(args.task_id),
        "num_episodes": int(args.num_episodes),
        "frames_per_episode": int(args.frames_per_episode),
        "layers": layers,
        "denoise_steps": denoise_steps,
        "num_inference_steps": int(num_inference_steps),
    }
    (out_root / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = task_suite.get_task(int(args.task_id))
    initial_states = task_suite.get_task_init_states(int(args.task_id))
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    prompt = DEFAULT_PROMPT.format(task=task_description)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 10))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 30))

    for episode_idx in range(int(args.num_episodes)):
        episode_dir = out_root / f"episode_{episode_idx:03d}"
        episode_stats: list[dict[str, Any]] = []
        episode_panels: dict[tuple[int, int], list[Path]] = {}
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
        for _ in range(num_steps_wait):
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done:
                break
        for frame_idx in range(int(args.frames_per_episode)):
            image, proprio, _ = _obs_to_model_input(
                obs,
                cfg=cfg,
                processor=processor,
                width=input_w,
                height=input_h,
                device=device,
                dtype=model.torch_dtype,
            )
            action, records = _infer_action_with_attention(
                model=model,
                model_type=args.model_type,
                prompt=prompt,
                input_image=image,
                proprio=proprio,
                action_horizon=action_horizon,
                num_inference_steps=num_inference_steps,
                sigma_shift=None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.sigma_shift),
                rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")),
                tiled=bool(cfg.EVALUATION.get("tiled", False)),
                layers=layers,
                denoise_steps=denoise_steps,
                seed=args.seed,
            )
            frame_dir = out_root / f"episode_{episode_idx:03d}" / f"frame_{frame_idx:03d}"
            frame_stats, panel_paths = _save_attention_outputs(
                frame_dir=frame_dir,
                model_type=args.model_type,
                episode_id=episode_idx,
                frame_id=frame_idx,
                model_rgb=_tensor_image_to_uint8(image),
                records=records,
                concat_multi_camera=concat_multi_camera,
            )
            episode_stats.extend(frame_stats)
            for key, panel_path in panel_paths.items():
                episode_panels.setdefault(key, []).append(panel_path)

            action_np = _denormalize_action(action, processor)[0]
            action_np[..., -1] = action_np[..., -1] * 2 - 1
            action_np = invert_gripper_action(action_np)
            if bool(cfg.EVALUATION.get("binarize_gripper", False)):
                action_np[..., -1] = np.sign(action_np[..., -1])
            done = False
            for env_action in action_np[:replan_steps].tolist():
                obs, _, done, _ = env.step(env_action)
                if done:
                    break
            if done:
                break
        if not args.no_video:
            for (layer, step), panel_paths in sorted(episode_panels.items()):
                _save_episode_video(
                    panel_paths,
                    episode_dir / f"attention_video_layer{layer}_step{step}.mp4",
                    fps=int(args.video_fps),
                )
        _save_episode_mass_trends(
            episode_dir=episode_dir,
            stats=episode_stats,
            model_type=args.model_type,
            representative_layer=layers[-1],
            representative_step=denoise_steps[-1],
        )
        stats_path = episode_dir / "episode_attention_stats.json"
        stats_path.write_text(json.dumps(episode_stats, indent=2), encoding="utf-8")
        print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()
