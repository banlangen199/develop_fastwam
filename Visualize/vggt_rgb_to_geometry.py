#!/usr/bin/env python3
"""Run the official VGGT model on RGB views and save depth and point clouds.

It accepts either ordinary image files or a Dream episode ``replan_XXXX.pt``
record containing ``record["rgb"]``. It loads VGGT through its official Python
package and writes raw NumPy predictions plus visualization products.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def _without_batch(value: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim < 2 or array.shape[0] != 1:
        raise ValueError(f"{name} must have a singleton batch dimension, got {array.shape}.")
    return array[0]


def _resolve_dtype(device: torch.device, requested: str) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        if device.type != "cuda":
            raise ValueError("float16 inference is only supported by this script on CUDA.")
        return torch.float16
    if requested == "bfloat16":
        if device.type != "cuda":
            raise ValueError("bfloat16 inference is only supported by this script on CUDA.")
        return torch.bfloat16
    if requested != "auto":
        raise ValueError(f"Unsupported dtype={requested!r}.")
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _processed_rgb(images: torch.Tensor) -> np.ndarray:
    """Convert official VGGT preprocessing output [S,3,H,W] to uint8 RGB."""
    array = images.detach().cpu().float().permute(0, 2, 3, 1).numpy()
    return (np.nan_to_num(array).clip(0, 1) * 255.0).round().astype(np.uint8)


def _uint8_rgb(value: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"{name} must be an RGB image with shape [H,W,3], got {array.shape}."
        )
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and float(np.nanmax(array)) <= 1.5:
            array *= 255.0
        array = np.nan_to_num(array).clip(0, 255).astype(np.uint8)
    return array


def _preprocess_rgb_arrays(rgb_arrays: list[np.ndarray]) -> torch.Tensor:
    """Apply the official VGGT crop preprocessing directly to in-memory RGB."""
    if not rgb_arrays:
        raise ValueError("At least one RGB view is required.")

    target_size = 518
    images: list[torch.Tensor] = []
    shapes: set[tuple[int, int]] = set()
    for array in rgb_arrays:
        image = Image.fromarray(array)
        width, height = image.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        image_tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            image_tensor = image_tensor[:, start_y : start_y + target_size, :]
        shapes.add((image_tensor.shape[1], image_tensor.shape[2]))
        images.append(image_tensor)

    if len(shapes) > 1:
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        padded_images = []
        for image in images:
            h_padding = max_height - image.shape[1]
            w_padding = max_width - image.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                image = torch.nn.functional.pad(
                    image,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded_images.append(image)
        images = padded_images

    return torch.stack(images)


def _load_record_images(
    record_path: Path,
    view_names: list[str],
) -> tuple[torch.Tensor, list[str]]:
    resolved = record_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Input record does not exist: {resolved}")
    record = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(record, dict) or not isinstance(record.get("rgb"), dict):
        raise ValueError(f"{resolved} must contain a record['rgb'] mapping.")
    missing = [name for name in view_names if name not in record["rgb"]]
    if missing:
        available = sorted(record["rgb"])
        raise KeyError(
            f"RGB views {missing} are missing from {resolved}; available views: {available}"
        )
    rgb_arrays = [
        _uint8_rgb(record["rgb"][name], name=f"record['rgb'][{name!r}]")
        for name in view_names
    ]
    images = _preprocess_rgb_arrays(rgb_arrays)
    inputs = [f"{resolved}::rgb.{name}" for name in view_names]
    return images, inputs


def _point_mask(
    points: np.ndarray,
    confidence: np.ndarray,
    *,
    confidence_percentile: float,
    stride: int,
) -> tuple[np.ndarray, float]:
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"points must be [S,H,W,3], got {points.shape}.")
    if confidence.shape != points.shape[:-1]:
        raise ValueError(
            f"confidence must match points [S,H,W], got {confidence.shape} "
            f"for points {points.shape}."
        )
    sampled_points = points[:, ::stride, ::stride]
    sampled_confidence = confidence[:, ::stride, ::stride]
    finite = np.isfinite(sampled_points).all(axis=-1) & np.isfinite(sampled_confidence)
    if not np.any(finite):
        raise ValueError("VGGT produced no finite points.")
    threshold = float(np.percentile(sampled_confidence[finite], confidence_percentile))
    return finite & (sampled_confidence >= threshold), threshold


def write_binary_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    """Write a binary little-endian colored point cloud."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {points.shape}.")
    if colors.shape != points.shape:
        raise ValueError(f"colors must match points [N,3], got {colors.shape}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


def _save_point_cloud(
    path: Path,
    points: np.ndarray,
    confidence: np.ndarray,
    rgb: np.ndarray,
    *,
    confidence_percentile: float,
    stride: int,
) -> dict[str, Any]:
    mask, threshold = _point_mask(
        points,
        confidence,
        confidence_percentile=confidence_percentile,
        stride=stride,
    )
    sampled_points = points[:, ::stride, ::stride]
    sampled_rgb = rgb[:, ::stride, ::stride]
    write_binary_ply(path, sampled_points[mask], sampled_rgb[mask])
    return {
        "path": str(path),
        "num_points": int(mask.sum()),
        "confidence_threshold": threshold,
        "confidence_percentile": float(confidence_percentile),
        "stride": int(stride),
    }


def _import_vggt(vggt_root: Path | None):
    if vggt_root is not None:
        resolved = vggt_root.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"--vggt-root does not exist: {resolved}")
        sys.path.insert(0, str(resolved))
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as error:
        raise ImportError(
            "The official VGGT package is not importable. Clone "
            "https://github.com/facebookresearch/vggt and run `pip install -e /path/to/vggt`, "
            "or pass `--vggt-root /path/to/vggt`."
        ) from error
    return (
        VGGT,
        load_and_preprocess_images,
        pose_encoding_to_extri_intri,
        unproject_depth_map_to_point_map,
    )


def run_vggt(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    from Visualize.modality_visualization import depth_colormap

    if args.images is not None and len(args.images) != len(args.view_names):
        raise ValueError("--view-names must contain exactly one name per --images entry.")
    if len(set(args.view_names)) != len(args.view_names):
        raise ValueError("--view-names entries must be unique.")
    if args.point_stride < 1:
        raise ValueError("--point-stride must be at least 1.")
    if not 0 <= args.confidence_percentile < 100:
        raise ValueError("--confidence-percentile must be in [0,100).")
    (
        VGGT,
        load_and_preprocess_images,
        pose_encoding_to_extri_intri,
        unproject_depth_map_to_point_map,
    ) = _import_vggt(args.vggt_root)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    dtype = _resolve_dtype(device, args.dtype)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.record is not None:
        images, input_sources = _load_record_images(args.record, args.view_names)
    else:
        image_paths = [path.expanduser().resolve() for path in args.images]
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Input image files do not exist: {missing}")
        images = load_and_preprocess_images([str(path) for path in image_paths])
        input_sources = [str(path) for path in image_paths]
    images = images.to(device)
    if images.ndim != 4 or images.shape[0] != len(args.view_names):
        raise ValueError(
            "Official VGGT preprocessing must return [S,3,H,W]; "
            f"got {tuple(images.shape)}."
        )
    rgb = _processed_rgb(images)
    batch_images = images.unsqueeze(0)

    model = VGGT.from_pretrained(args.model_id).to(device).eval()
    with torch.inference_mode():
        with _autocast_context(device, dtype):
            aggregated_tokens, patch_start_index = model.aggregator(batch_images)
            pose_encoding = model.camera_head(aggregated_tokens)[-1]
            depth, depth_confidence = model.depth_head(
                aggregated_tokens, batch_images, patch_start_index
            )
            point_map, point_confidence = model.point_head(
                aggregated_tokens, batch_images, patch_start_index
            )
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_encoding, batch_images.shape[-2:]
        )

    depth_with_channel = _without_batch(depth, name="depth")
    depth_confidence = _without_batch(depth_confidence, name="depth_confidence")
    point_map = _without_batch(point_map, name="point_map")
    point_confidence = _without_batch(point_confidence, name="point_confidence")
    extrinsic = _without_batch(extrinsic, name="extrinsic")
    intrinsic = _without_batch(intrinsic, name="intrinsic")
    pose_encoding = _without_batch(pose_encoding, name="pose_encoding")

    if depth_with_channel.ndim != 4 or depth_with_channel.shape[-1] != 1:
        raise ValueError(
            f"VGGT depth must be [S,H,W,1], got {depth_with_channel.shape}."
        )
    depth = depth_with_channel[..., 0]
    world_points_from_depth = np.asarray(
        unproject_depth_map_to_point_map(
            depth_with_channel,
            extrinsic,
            intrinsic,
        ),
        dtype=np.float32,
    )

    np.savez_compressed(
        output_dir / "predictions.npz",
        view_names=np.asarray(args.view_names),
        input_paths=np.asarray(input_sources),
        processed_rgb=rgb,
        pose_encoding=pose_encoding,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth=depth,
        depth_confidence=depth_confidence,
        point_map=point_map,
        point_confidence=point_confidence,
        world_points_from_depth=world_points_from_depth,
    )

    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    depth_range = (
        [float(x) for x in np.percentile(valid_depth, [2.0, 98.0])]
        if valid_depth.size
        else [0.0, 1.0]
    )
    for index, view_name in enumerate(args.view_names):
        cv2.imwrite(
            str(depth_dir / f"{view_name}.png"),
            cv2.cvtColor(
                depth_colormap(depth[index], value_range=tuple(depth_range)),
                cv2.COLOR_RGB2BGR,
            ),
        )
        cv2.imwrite(
            str(depth_dir / f"{view_name}_input.png"),
            cv2.cvtColor(rgb[index], cv2.COLOR_RGB2BGR),
        )

    point_clouds = {
        "from_depth": _save_point_cloud(
            output_dir / "pointcloud_from_depth.ply",
            world_points_from_depth,
            depth_confidence,
            rgb,
            confidence_percentile=args.confidence_percentile,
            stride=args.point_stride,
        ),
        "from_point_head": _save_point_cloud(
            output_dir / "pointcloud_from_point_head.ply",
            point_map,
            point_confidence,
            rgb,
            confidence_percentile=args.confidence_percentile,
            stride=args.point_stride,
        ),
    }
    manifest = {
        "model_id": args.model_id,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "view_names": list(args.view_names),
        "input_paths": input_sources,
        "input_record": str(args.record.expanduser().resolve()) if args.record else None,
        "processed_shape": list(rgb.shape),
        "depth_range_percentile_2_98": depth_range,
        "raw_predictions": str(output_dir / "predictions.npz"),
        "depth_visualizations": str(depth_dir),
        "point_clouds": point_clouds,
        "notes": [
            "extrinsic uses VGGT/OpenCV camera-from-world convention",
            "pointcloud_from_depth uses VGGT depth, predicted cameras, and official unprojection",
            "VGGT geometry is not guaranteed to share metric scale or world frame with FastWAM targets",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--images",
        type=Path,
        nargs="+",
        help="Ordered RGB image paths, normally primary then wrist.",
    )
    inputs.add_argument(
        "--record",
        type=Path,
        help="Dream episode replan_XXXX.pt containing record['rgb'].",
    )
    parser.add_argument(
        "--view-names",
        nargs="+",
        default=["image", "wrist_image"],
        help=(
            "Ordered output view names. With --record these are also the keys "
            "read from record['rgb'] (default: image wrist_image)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-id",
        default="facebook/VGGT-1B",
        help="Hugging Face model id or local from_pretrained directory.",
    )
    parser.add_argument(
        "--vggt-root",
        type=Path,
        default=None,
        help="Optional official VGGT repository root if it is not installed.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=20.0,
        help="Drop points below this confidence percentile (default: 20).",
    )
    parser.add_argument(
        "--point-stride",
        type=int,
        default=2,
        help="Keep every Nth point in both image dimensions (default: 2).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_vggt(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
