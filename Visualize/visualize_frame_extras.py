#!/usr/bin/env python
"""Visualize RGB + extras for a single frame, including both front and wrist cameras.

Renders a multi-panel grid layout:

  ┌──────────────┬──────────────┐
  │  Front RGB   │  Wrist RGB   │
  ├──────────────┼──────────────┤
  │ F dyn heat   │ W dyn heat   │
  ├──────────────┼──────────────┤
  │ F  DA3 Depth │ W  DA3 Depth │
  ├──────────────┼──────────────┤
  │ F SAM region │ W SAM region │
  ├──────────────┼──────────────┤
  │ F DINO PCA   │ W DINO PCA   │
  └──────────────┴──────────────┘

Usage:
  # Single frame
  python Visualize/visualize_frame_extras.py \
    --dataset libero_goal --episode 0 --frame 50 --output Visualize/results/frame_ep0_f50.png

  # Entire episode as video
  python Visualize/visualize_frame_extras.py \
    --dataset libero_goal --episode 0 --output Visualize/results/episode_0.mp4 --fps 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import cv2
from tqdm import tqdm

from modality_visualization import (
    as_dense_features,
    depth_colormap,
    dynamic_map_from_tracks,
    fit_kmeans_projection,
    fit_pca_projection,
    overlay_dynamic_heatmap,
    overlay_embedding_regions,
    overlay_feature_pca,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEROBOT_ROOT = PROJECT_ROOT / "data" / "libero_mujoco3.3.2"

# Camera display names
CAMERA_DISPLAY = {
    "image": "Front Camera",
    "wrist_image": "Wrist Camera",
}

# Available extras with display names and their npz keys
EXTRAS_SPEC = {
    "cotracker": {
        "display": "Dynamics heatmap",
        "array_key": "tracks",
        "visualizer": "dynamic",
        "spatial_hw": (28, 28),
    },
    "depth_anything_v3_metric": {
        "display": "Depth Anything V3",
        "array_key": "depth",
        "visualizer": "depth",
    },
    "sam": {
        "display": "SAM regions",
        "array_key": "features",
        "visualizer": "sam_regions",
        "spatial_hw": (16, 16),
    },
    "dinov2": {
        "display": "DINOv2 PCA",
        "array_key": "features",
        "visualizer": "dino_pca",
        "spatial_hw": (16, 16),
    },
}


def lerobot_to_uint8_rgb(frames: torch.Tensor) -> np.ndarray:
    """Convert LeRobot video frames [C, H, W] float [0,1] → [H, W, 3] uint8 RGB."""
    f = frames.detach().cpu().float()
    if f.ndim == 4:
        f = f[0]  # take first if batched
    # frames are in [0, 1] from LeRobot
    f = (f * 255.0).clamp(0, 255).to(torch.uint8)
    return f.permute(1, 2, 0).contiguous().numpy()


def add_label(img: np.ndarray, text: str, position: str = "top") -> np.ndarray:
    """Add a text label to an image."""
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    color = (255, 255, 255)
    margin = 4

    if position == "top":
        y = margin + 10
        x = margin
    else:
        y = out.shape[0] - margin - 4
        x = margin

    # Dark background for readability
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(out, (x - 2, y - th - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)
    cv2.putText(out, text, (x, y), font, font_scale, color, thickness)
    return out


def load_extras_frame(
    extras_root: Path,
    feature_name: str,
    extras_spec: dict,
    camera: str,
    episode: int,
    frame_idx: int,
    rgb: np.ndarray,
    *,
    overlay_alpha: float = 0.48,
    sam_clusters: int = 8,
    draw_contours: bool = True,
    visual_context: dict | None = None,
) -> np.ndarray | None:
    """Load one frame of an extra feature as uint8 RGB [H, W, 3].

    Returns None if the feature is not available for this camera/frame.
    """
    p = extras_root / feature_name / camera / f"episode_{episode:06d}.npz"
    if not p.exists():
        return None

    data = np.load(p, allow_pickle=True)
    spec = extras_spec[feature_name]

    arr_key = spec["array_key"]
    if frame_idx >= len(data[arr_key]):
        return None
    visualizer = spec["visualizer"]
    if visualizer == "sam_regions":
        result = overlay_embedding_regions(
            rgb,
            data[arr_key][frame_idx],
            spec["spatial_hw"],
            clusters=sam_clusters,
            alpha=overlay_alpha,
            draw_contours=draw_contours,
            projection=(visual_context or {}).get("sam_projection"),
        )
    elif visualizer == "dynamic":
        visibility = data["visibility"][frame_idx] if "visibility" in data else None
        motion = dynamic_map_from_tracks(
            data[arr_key][frame_idx],
            visibility,
            grid_hw=spec["spatial_hw"],
        )
        result = overlay_dynamic_heatmap(
            rgb, motion, alpha=overlay_alpha,
            scale=(visual_context or {}).get("dynamic_scale"),
        )
    elif visualizer == "dino_pca":
        result = overlay_feature_pca(
            rgb,
            data[arr_key][frame_idx],
            spec["spatial_hw"],
            alpha=overlay_alpha,
            projection=(visual_context or {}).get("dino_projection"),
        )
    else:
        depth_frame = data[arr_key][frame_idx].astype(np.float32)
        result = depth_colormap(depth_frame)

    return result


def build_episode_visual_context(
    extras_root: Path,
    camera: str,
    episode: int,
    *,
    sam_clusters: int,
    sample_frames: int = 8,
) -> dict:
    """Fit stable SAM/DINO colors and a dyn scale once for an episode/camera."""
    context: dict = {}

    sam_path = extras_root / "sam" / camera / f"episode_{episode:06d}.npz"
    if sam_path.exists():
        with np.load(sam_path, allow_pickle=False) as data:
            features = data["features"]
            indices = np.linspace(0, len(features) - 1, min(sample_frames, len(features)), dtype=int)
            dense = np.concatenate(
                [as_dense_features(features[index], (16, 16)).reshape(-1, 256) for index in indices],
                axis=0,
            )
        context["sam_projection"] = fit_kmeans_projection(dense, clusters=sam_clusters)

    dino_path = extras_root / "dinov2" / camera / f"episode_{episode:06d}.npz"
    if dino_path.exists():
        with np.load(dino_path, allow_pickle=False) as data:
            features = data["features"]
            indices = np.linspace(0, len(features) - 1, min(sample_frames, len(features)), dtype=int)
            dense = np.concatenate(
                [as_dense_features(features[index], (16, 16)).reshape(-1, features.shape[-1])
                 for index in indices],
                axis=0,
            )
        context["dino_projection"] = fit_pca_projection(dense)

    track_path = extras_root / "cotracker" / camera / f"episode_{episode:06d}.npz"
    if track_path.exists():
        with np.load(track_path, allow_pickle=False) as data:
            magnitude = np.linalg.norm(data["tracks"].astype(np.float32), axis=-1)
            if "visibility" in data:
                magnitude *= np.clip(data["visibility"].astype(np.float32), 0, 1)
            positive = magnitude[np.isfinite(magnitude) & (magnitude > 0)]
            context["dynamic_scale"] = (
                float(np.percentile(positive, 95.0)) if positive.size else 1.0
            )
    return context


def render_frame_grid(
    ds,
    extras_root: Path,
    extras_spec: dict,
    episode: int,
    frame_idx: int,
    cameras: list[str],
    panel_size: int,
    border: int = 2,
    show_frame_info: bool = True,
    overlay_alpha: float = 0.48,
    sam_clusters: int = 8,
    draw_contours: bool = True,
    visual_contexts: dict[str, dict] | None = None,
) -> np.ndarray:
    """Render a single frame grid with all cameras and extras.

    Returns [H, W, 3] uint8 BGR array (ready for cv2 imwrite/videowriter).
    """
    n_cameras = len(cameras)
    n_rows = 1 + len(extras_spec)  # RGB row + one row per extra
    n_cols = n_cameras

    canvas_h = n_rows * panel_size + (n_rows + 1) * border
    canvas_w = n_cols * panel_size + (n_cols + 1) * border
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Background fill
    canvas[:] = (30, 30, 30)  # dark gray

    # Get episode timestamps for video query
    ep_start = ds.episode_data_index["from"][episode].item()
    ep_end = ds.episode_data_index["to"][episode].item()
    rows_data = ds.hf_dataset[list(range(ep_start, ep_end))]
    timestamps = [float(t.item() if hasattr(t, "item") else t)
                  for t in rows_data["timestamp"]]

    # Camera alias mapping
    camera_aliases = {
        "image": "observation.images.image",
        "wrist_image": "observation.images.wrist_image",
    }

    for col_idx, camera in enumerate(cameras):
        video_key = camera_aliases.get(camera, f"observation.images.{camera}")
        try:
            vf = ds._query_videos({video_key: [timestamps[frame_idx]]}, episode)
            camera_rgb = lerobot_to_uint8_rgb(vf[video_key])
        except Exception as exc:
            camera_rgb = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
            camera_rgb = add_label(camera_rgb, f"RGB error: {exc}", "top")

        for row_idx in range(n_rows):
            y0 = border + row_idx * (panel_size + border)
            x0 = border + col_idx * (panel_size + border)

            if row_idx == 0:
                panel = cv2.resize(camera_rgb, (panel_size, panel_size))
                label = f"{CAMERA_DISPLAY.get(camera, camera)}  RGB"
            else:
                # Extras row
                feature_names = list(extras_spec.keys())
                feat_name = feature_names[row_idx - 1]
                feat_display = extras_spec[feat_name]["display"]

                panel = load_extras_frame(
                    extras_root,
                    feat_name,
                    extras_spec,
                    camera,
                    episode,
                    frame_idx,
                    camera_rgb,
                    overlay_alpha=overlay_alpha,
                    sam_clusters=sam_clusters,
                    draw_contours=draw_contours,
                    visual_context=(visual_contexts or {}).get(camera),
                )
                if panel is None:
                    panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
                    panel = add_label(panel, f"{feat_display}: N/A", "top")
                else:
                    panel = cv2.resize(panel, (panel_size, panel_size))

                label = f"{CAMERA_DISPLAY.get(camera, camera)}  {feat_display}"

            panel = add_label(panel, label)
            panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            canvas[y0:y0 + panel_size, x0:x0 + panel_size] = panel

    # Add global frame info at bottom if there's room
    if show_frame_info:
        info_text = f"Episode {episode}  |  Frame {frame_idx}"
        y_info = canvas_h - 5
        cv2.putText(canvas, info_text, (5, y_info),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="LIBERO dataset suite name",
    )
    parser.add_argument(
        "--episode", type=int, default=0,
        help="Episode index to visualize",
    )
    parser.add_argument(
        "--frame", type=int, default=0,
        help="Frame index within the episode (0-based)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path (.png for single frame, .mp4 for video); "
             "default: evaluate_results/visualizations/{dataset}_ep{ep}_f{frame}.png",
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="FPS when rendering a video (only when --output is .mp4)",
    )
    parser.add_argument(
        "--panel-size", type=int, default=300,
        help="Each panel H,W in pixels (default: 300)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit frames when rendering video (for testing)",
    )
    parser.add_argument(
        "--cameras", type=str, nargs="+", default=["image", "wrist_image"],
        help="Cameras to include (default: image wrist_image)",
    )
    parser.add_argument(
        "--no-extras", action="store_true",
        help="Only render RGB, skip all extras",
    )
    parser.add_argument(
        "--all-frames", action="store_true",
        help="Render all frames as video instead of a single frame",
    )
    parser.add_argument(
        "--overlay-alpha", type=float, default=0.48,
        help="Opacity for SAM/dynamics/DINO overlays (default: 0.48)",
    )
    parser.add_argument(
        "--sam-clusters", type=int, default=8,
        help="Number of KMeans regions for stored SAM embeddings (default: 8)",
    )
    parser.add_argument(
        "--no-contours", action="store_true",
        help="Disable contours on SAM embedding regions",
    )
    args = parser.parse_args()

    # Add project root to path for imports
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    suite_name = args.dataset
    lerobot_dir = LEROBOT_ROOT / f"{suite_name}_no_noops_lerobot"

    if not lerobot_dir.exists():
        print(f"Dataset not found: {lerobot_dir}")
        sys.exit(1)

    print(f"Loading dataset: {lerobot_dir}")
    ds = LeRobotDataset(repo_id=str(lerobot_dir), root=lerobot_dir, download_videos=False)

    ep_idx = args.episode
    ep_len = ds.meta.episodes[ep_idx]["length"]
    print(f"Episode {ep_idx}: {ep_len} frames")

    extras_root = lerobot_dir / "extras"
    if not args.no_extras and not extras_root.exists():
        print(f"Warning: extras not found at {extras_root}, showing RGB only")
        args.no_extras = True

    # Build extras spec for this run
    active_extras = {} if args.no_extras else dict(EXTRAS_SPEC)
    visual_contexts = {}
    if not args.no_extras:
        print("Fitting episode-level color mappings (stable across frames)...")
        for camera in args.cameras:
            visual_contexts[camera] = build_episode_visual_context(
                extras_root,
                camera,
                ep_idx,
                sam_clusters=args.sam_clusters,
            )

    # Determine output path
    if args.all_frames or (args.output and args.output.suffix == ".mp4"):
        # Video mode
        out_path = args.output or (
            PROJECT_ROOT / "evaluate_results" / "visualizations"
            / f"{suite_name}_cameras_ep{ep_idx:04d}.mp4"
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        total_frames = min(ep_len, args.max_frames or ep_len)
        P = args.panel_size
        n_rows = 1 + len(active_extras)
        n_cols = len(args.cameras)
        border = 2
        canvas_h = n_rows * P + (n_rows + 1) * border
        canvas_w = n_cols * P + (n_cols + 1) * border

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (canvas_w, canvas_h))

        print(f"Rendering episode {ep_idx} ({total_frames} frames) → {out_path}")
        for frm in tqdm(range(total_frames), desc="frames"):
            frame_canvas = render_frame_grid(
                ds, extras_root, active_extras, ep_idx, frm, args.cameras, P,
                show_frame_info=True,
                overlay_alpha=args.overlay_alpha,
                sam_clusters=args.sam_clusters,
                draw_contours=not args.no_contours,
                visual_contexts=visual_contexts,
            )
            writer.write(frame_canvas)

        writer.release()
        print(f"Saved video to {out_path}")

    else:
        # Single frame mode
        frame_idx = args.frame
        if frame_idx >= ep_len:
            print(f"Error: frame {frame_idx} >= episode length {ep_len}")
            sys.exit(1)

        out_path = args.output or (
            PROJECT_ROOT / "evaluate_results" / "visualizations"
            / f"{suite_name}_ep{ep_idx:04d}_f{frame_idx:04d}.png"
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Rendering episode {ep_idx}, frame {frame_idx} → {out_path}")
        frame_canvas = render_frame_grid(
            ds, extras_root, active_extras, ep_idx, frame_idx, args.cameras,
            args.panel_size, show_frame_info=True,
            overlay_alpha=args.overlay_alpha,
            sam_clusters=args.sam_clusters,
            draw_contours=not args.no_contours,
            visual_contexts=visual_contexts,
        )

        cv2.imwrite(str(out_path), frame_canvas)
        print(f"Saved image to {out_path}")


if __name__ == "__main__":
    main()
