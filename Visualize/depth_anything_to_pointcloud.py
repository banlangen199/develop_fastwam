#!/usr/bin/env python3
"""Back-project Depth Anything metric depth to a PLY point cloud.

Depth can come from dataset extras or directly from a DreamFastWAM
``replan_XXXX.pt`` prediction record. For example:

  python Visualize/depth_anything_to_pointcloud.py \
    --dataset-root data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot \
    --episode 0 --frame 0 --output pointcloud.ply

  python Visualize/depth_anything_to_pointcloud.py \
    --record raw_predictions/replan_0000.pt \
    --future-offset 16 --cameras image --no-color \
    --output dream_depth_t16_image.ply

Depth Anything V3 metric depth is interpreted as Z depth in metres.  LIBERO's
default camera vertical field of view is 45 degrees.  Pass calibrated
intrinsics with ``--intrinsics fx fy cx cy`` when available.

Two cameras can be fused by passing ``--cameras image wrist_image`` and one
camera-to-world pose per camera.  A pose file can contain a [4, 4] matrix, or a
[T, 4, 4] array for a moving camera (notably the wrist camera):

  python Visualize/depth_anything_to_pointcloud.py ... \
    --cameras image wrist_image \
    --pose image=agent_camera_to_world.npy \
    --pose wrist_image=wrist_camera_to_world.npy
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


VIDEO_KEYS = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
}


def load_depth(extras_root: Path, camera: str, episode: int, frame: int) -> np.ndarray:
    """Load one depth map, selecting by stored frame_index rather than row."""
    stem = extras_root / "depth_anything_v3_metric" / camera / f"episode_{episode:06d}"
    npz_path = stem.with_suffix(".npz")
    if npz_path.exists():
        with np.load(npz_path) as data:
            indices = data["frame_index"] if "frame_index" in data else np.arange(len(data["depth"]))
            matches = np.flatnonzero(indices == frame)
            if not len(matches):
                raise ValueError(f"frame {frame} is absent from {npz_path}")
            row = int(matches[0])
            if "valid" in data and not bool(data["valid"][row]):
                raise ValueError(f"frame {frame} is marked invalid in {npz_path}")
            return np.asarray(data["depth"][row], dtype=np.float32)

    # Support datasets converted from NPZ to the split NPY representation.
    depth_path = Path(f"{stem}.depth.npy")
    index_path = Path(f"{stem}.frame_index.npy")
    valid_path = Path(f"{stem}.valid.npy")
    if not depth_path.exists():
        raise FileNotFoundError(f"neither {npz_path} nor {depth_path} exists")
    indices = np.load(index_path, mmap_mode="r") if index_path.exists() else np.arange(len(np.load(depth_path, mmap_mode="r")))
    matches = np.flatnonzero(indices == frame)
    if not len(matches):
        raise ValueError(f"frame {frame} is absent from {depth_path}")
    row = int(matches[0])
    if valid_path.exists() and not bool(np.load(valid_path, mmap_mode="r")[row]):
        raise ValueError(f"frame {frame} is marked invalid in {valid_path}")
    return np.asarray(np.load(depth_path, mmap_mode="r")[row], dtype=np.float32)


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().float().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _split_flat_square_views(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if prediction.ndim != 2 or prediction.shape[0] % 2 != 0:
        raise ValueError(
            "Dream depth must contain two flattened views with shape [2N,P], "
            f"got {prediction.shape}."
        )
    per_view = prediction.shape[0] // 2
    side = int(round(per_view**0.5))
    if side * side != per_view:
        raise ValueError(
            f"Each Dream depth view must form a square grid, got {per_view} positions."
        )
    combined = prediction.reshape(side, side * 2, prediction.shape[-1])
    return combined[:, :side], combined[:, side:]


def _unpatchify_depth(patch_grid: np.ndarray) -> np.ndarray:
    if patch_grid.ndim != 3:
        raise ValueError(f"Dream depth patch grid must be [H,W,P], got {patch_grid.shape}.")
    patch_side = int(round(patch_grid.shape[-1] ** 0.5))
    if patch_side * patch_side != patch_grid.shape[-1]:
        raise ValueError(
            f"Dream depth patch dimension must be square, got {patch_grid.shape[-1]}."
        )
    grid_h, grid_w = patch_grid.shape[:2]
    return (
        patch_grid.reshape(grid_h, grid_w, patch_side, patch_side)
        .transpose(0, 2, 1, 3)
        .reshape(grid_h * patch_side, grid_w * patch_side)
        .astype(np.float32, copy=False)
    )


def load_record_depths(
    record_path: Path,
    *,
    future_offset: int | None,
    horizon_index: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], int, int]:
    """Load and unpatchify both predicted depth views from a Dream record."""
    import torch

    resolved = record_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dream prediction record does not exist: {resolved}")
    record = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(record, dict):
        raise ValueError(f"Dream prediction record must be a mapping: {resolved}")
    predictions = record.get("dream_predictions")
    if not isinstance(predictions, dict) or "depth" not in predictions:
        raise ValueError(f"{resolved} does not contain dream_predictions['depth'].")

    depth_predictions = _as_numpy(predictions["depth"])
    offsets = [int(value) for value in record.get("future_offsets", [])]
    if depth_predictions.ndim != 3:
        raise ValueError(
            "dream_predictions['depth'] must be [num_horizons,2N,P], "
            f"got {depth_predictions.shape}."
        )
    if len(offsets) != depth_predictions.shape[0]:
        raise ValueError(
            f"future_offsets={offsets} does not match depth horizons "
            f"{depth_predictions.shape[0]}."
        )

    if future_offset is not None:
        if future_offset not in offsets:
            raise ValueError(
                f"future offset {future_offset} is unavailable; choose one of {offsets}."
            )
        selected_index = offsets.index(future_offset)
    else:
        selected_index = 0 if horizon_index is None else horizon_index
        if not 0 <= selected_index < len(offsets):
            raise IndexError(
                f"horizon index {selected_index} is outside [0,{len(offsets) - 1}]."
            )
    selected_offset = offsets[selected_index]

    primary_grid, wrist_grid = _split_flat_square_views(
        np.asarray(depth_predictions[selected_index], dtype=np.float32)
    )
    depths = {
        "image": _unpatchify_depth(primary_grid),
        "wrist_image": _unpatchify_depth(wrist_grid),
    }
    return depths, record, selected_index, selected_offset


def intrinsics_from_fovy(height: int, width: int, fovy_degrees: float) -> tuple[float, float, float, float]:
    """Return fx, fy, cx, cy for square pixels and a vertical field of view."""
    fy = 0.5 * height / math.tan(math.radians(fovy_degrees) / 2.0)
    return fy, fy, (width - 1.0) / 2.0, (height - 1.0) / 2.0


def backproject(
    depth: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    stride: int = 1,
    min_depth: float = 0.0,
    max_depth: float = math.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a depth image; return points and their flattened pixel ids."""
    if depth.ndim != 2:
        raise ValueError(f"expected [H,W] depth, got {depth.shape}")
    fx, fy, cx, cy = intrinsics
    if fx <= 0 or fy <= 0:
        raise ValueError("fx and fy must be positive")
    vv, uu = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride]
    keep = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    z, u, v = z[keep], uu[keep], vv[keep]
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    pixel_ids = (v * depth.shape[1] + u).astype(np.int64)
    return points.astype(np.float32), pixel_ids


def load_pose(path: Path, frame: int) -> np.ndarray:
    """Load a camera-to-world transform from NPY/NPZ/JSON/TXT."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        pose = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            key = "camera_to_world" if "camera_to_world" in data else data.files[0]
            pose = np.asarray(data[key])
    elif suffix == ".json":
        pose = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)
    else:
        pose = np.loadtxt(path)
    if pose.ndim == 3:
        if frame >= len(pose):
            raise IndexError(f"pose file {path} has {len(pose)} frames, requested {frame}")
        pose = pose[frame]
    if pose.shape != (4, 4):
        raise ValueError(f"camera-to-world pose in {path} must be [4,4] or [T,4,4], got {pose.shape}")
    return np.asarray(pose, dtype=np.float64)


def transform_points(points: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    return (points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]).astype(np.float32)


def read_video_frame(video: Path, frame: int, output_hw: tuple[int, int]) -> np.ndarray:
    """Decode one RGB frame with ffmpeg and resize it to the depth resolution."""
    height, width = output_hw
    vf = f"select=eq(n\\,{frame}),scale={width}:{height}"
    command = [
        "ffmpeg", "-v", "error", "-i", str(video), "-vf", vf, "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    expected = height * width * 3
    if len(result.stdout) != expected:
        raise RuntimeError(f"ffmpeg returned {len(result.stdout)} bytes for {video}; expected {expected}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)


def resize_record_rgb(value: Any, output_hw: tuple[int, int]) -> np.ndarray:
    """Resize the record's current-observation RGB to the depth resolution."""
    rgb = _as_numpy(value)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"record RGB must be [H,W,3], got {rgb.shape}.")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        if rgb.size and float(np.nanmax(rgb)) <= 1.5:
            rgb *= 255.0
        rgb = np.nan_to_num(rgb).clip(0, 255).astype(np.uint8)
    height, width = output_hw
    return np.asarray(
        Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR)
    )


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write an interoperable ASCII PLY without requiring Open3D."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if colors is not None and (colors.shape != points.shape or colors.dtype != np.uint8):
        raise ValueError("colors must be uint8 [N,3]")
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        if colors is None:
            np.savetxt(stream, points, fmt="%.7g %.7g %.7g")
        else:
            rows = np.column_stack((points, colors))
            np.savetxt(stream, rows, fmt="%.7g %.7g %.7g %d %d %d")


def parse_assignments(values: list[str]) -> Mapping[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected CAMERA=PATH, got {value!r}")
        camera, path = value.split("=", 1)
        result[camera] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", type=Path)
    source.add_argument(
        "--record",
        type=Path,
        help="DreamFastWAM replan_XXXX.pt containing predicted depth.",
    )
    parser.add_argument("--episode", type=int)
    parser.add_argument("--frame", type=int, help="LeRobot frame_index within the episode")
    horizon = parser.add_mutually_exclusive_group()
    horizon.add_argument(
        "--future-offset",
        type=int,
        help="Select a Dream prediction by its future offset, e.g. 16 or 32.",
    )
    horizon.add_argument(
        "--horizon-index",
        type=int,
        help="Select a Dream prediction by zero-based index (default: 0).",
    )
    parser.add_argument("--cameras", nargs="+", choices=tuple(VIDEO_KEYS), default=["image"])
    parser.add_argument("--output", type=Path, required=True)
    intrinsics_group = parser.add_mutually_exclusive_group()
    intrinsics_group.add_argument("--fovy", type=float, default=45.0, help="vertical field of view in degrees (default: 45)")
    intrinsics_group.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--pose", action="append", default=[], metavar="CAMERA=PATH",
                        help="camera-to-world [4,4] or [T,4,4]; required when fusing cameras")
    parser.add_argument(
        "--pose-frame",
        type=int,
        help=(
            "Frame selected from a [T,4,4] pose file for --record. Defaults to "
            "record metadata env_step plus the selected future offset."
        ),
    )
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth pixel")
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.dataset_root is not None:
        if args.episode is None or args.frame is None:
            parser.error("--dataset-root requires --episode and --frame")
        if args.future_offset is not None or args.horizon_index is not None:
            parser.error("--future-offset/--horizon-index can only be used with --record")
        if args.pose_frame is not None:
            parser.error("--pose-frame can only be used with --record")
        record_depths = None
        record = None
        selected_offset = None
        pose_frame = args.frame
    else:
        if args.episode is not None or args.frame is not None:
            parser.error("--episode/--frame can only be used with --dataset-root")
        try:
            record_depths, record, _, selected_offset = load_record_depths(
                args.record,
                future_offset=args.future_offset,
                horizon_index=args.horizon_index,
            )
        except (FileNotFoundError, IndexError, KeyError, ValueError) as error:
            parser.error(str(error))
        metadata = record.get("metadata", {})
        pose_frame = args.pose_frame
        if pose_frame is None:
            pose_frame = int(metadata.get("env_step", 0)) + selected_offset

    poses = parse_assignments(args.pose)
    if len(args.cameras) > 1:
        missing = [camera for camera in args.cameras if camera not in poses]
        if missing:
            parser.error("multi-camera fusion requires --pose for every camera; missing: " + ", ".join(missing))

    all_points, all_colors = [], []
    for camera in args.cameras:
        if args.dataset_root is not None:
            depth = load_depth(
                args.dataset_root / "extras", camera, args.episode, args.frame
            )
        else:
            depth = record_depths[camera]
        intrinsics = tuple(args.intrinsics) if args.intrinsics else intrinsics_from_fovy(*depth.shape, args.fovy)
        points, pixel_ids = backproject(depth, intrinsics, args.stride, args.min_depth, args.max_depth)
        if camera in poses:
            points = transform_points(points, load_pose(poses[camera], pose_frame))
        all_points.append(points)

        if not args.no_color:
            if args.dataset_root is not None:
                video = (
                    args.dataset_root / "videos" / "chunk-000" / VIDEO_KEYS[camera]
                    / f"episode_{args.episode:06d}.mp4"
                )
                rgb = read_video_frame(video, args.frame, depth.shape)
            else:
                rgb_views = record.get("rgb")
                if not isinstance(rgb_views, dict) or camera not in rgb_views:
                    parser.error(
                        f"--record has no current-observation RGB for camera {camera!r}; "
                        "pass --no-color"
                    )
                rgb = resize_record_rgb(rgb_views[camera], depth.shape)
            all_colors.append(rgb.reshape(-1, 3)[pixel_ids])

    points = np.concatenate(all_points, axis=0)
    if not len(points):
        parser.error("No depth samples remain after filtering.")
    colors = None if args.no_color else np.concatenate(all_colors, axis=0)
    write_ply(args.output, points, colors)
    bounds_min, bounds_max = points.min(axis=0), points.max(axis=0)
    print(f"Wrote {len(points):,} points to {args.output}")
    print(f"bounds min={bounds_min.tolist()} max={bounds_max.tolist()} (metres)")
    if args.record is not None:
        print(f"Dream depth horizon: t+{selected_offset}")
        if not args.no_color:
            print(
                "Warning: point colors come from the current observation RGB, "
                "not a future RGB prediction."
            )
    if len(args.cameras) == 1 and args.cameras[0] not in poses:
        print("Coordinate frame: camera (x right, y down, z forward)")
    else:
        print("Coordinate frame: that of the supplied camera-to-world pose(s)")


if __name__ == "__main__":
    main()
