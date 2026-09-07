from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from Visualize.modality_visualization import (
    depth_colormap,
    fit_kmeans_projection,
    fit_pca_projection,
    overlay_dynamic_heatmap,
    overlay_embedding_regions,
    overlay_feature_pca,
)


CAMERAS = ("image", "wrist_image")


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def _uint8_rgb(image: Any) -> np.ndarray:
    array = _as_numpy(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"RGB image must be [H,W,3], got {array.shape}.")
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and float(np.nanmax(array)) <= 1.5:
            array *= 255.0
        array = np.nan_to_num(array).clip(0, 255).astype(np.uint8)
    return array


def _split_dense_grid(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if prediction.ndim != 3 or prediction.shape[1] % 2 != 0:
        raise ValueError(
            f"Expected a horizontally concatenated [H,2W,C] prediction, got {prediction.shape}."
        )
    half = prediction.shape[1] // 2
    return prediction[:, :half], prediction[:, half:]


def _split_flat_square_views(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if prediction.ndim != 2 or prediction.shape[0] % 2 != 0:
        raise ValueError(
            f"Expected a two-view flattened [2N,C] prediction, got {prediction.shape}."
        )
    per_view = prediction.shape[0] // 2
    side = int(round(per_view**0.5))
    if side * side != per_view:
        raise ValueError(
            f"Each camera prediction must form a square grid, got {per_view} positions."
        )
    combined = prediction.reshape(side, side * 2, prediction.shape[-1])
    return combined[:, :side], combined[:, side:]


def _unpatchify_depth(patch_grid: np.ndarray) -> np.ndarray:
    if patch_grid.ndim != 3:
        raise ValueError(f"Depth patch grid must be [H,W,P], got {patch_grid.shape}.")
    patch_side = int(round(patch_grid.shape[-1] ** 0.5))
    if patch_side * patch_side != patch_grid.shape[-1]:
        raise ValueError(
            f"Depth patch feature dimension must be square, got {patch_grid.shape[-1]}."
        )
    grid_h, grid_w = patch_grid.shape[:2]
    return (
        patch_grid.reshape(grid_h, grid_w, patch_side, patch_side)
        .transpose(0, 2, 1, 3)
        .reshape(grid_h * patch_side, grid_w * patch_side)
    )


def split_prediction_views(
    modality: str,
    prediction: Any,
) -> dict[str, np.ndarray]:
    """Split one horizon prediction into primary and wrist camera arrays."""
    array = _as_numpy(prediction)
    if modality in {"dino", "sam"}:
        primary, wrist = _split_dense_grid(array)
    elif modality == "depth":
        if array.ndim == 2 and array.shape[1] == 2 * array.shape[0]:
            half = array.shape[1] // 2
            primary, wrist = array[:, :half], array[:, half:]
        else:
            # Backward compatibility for prediction records produced before
            # depth decoders returned full maps.
            primary_grid, wrist_grid = _split_flat_square_views(array)
            primary = _unpatchify_depth(primary_grid)
            wrist = _unpatchify_depth(wrist_grid)
    elif modality == "dyn":
        primary_grid, wrist_grid = _split_flat_square_views(array)
        primary = primary_grid.squeeze(-1)
        wrist = wrist_grid.squeeze(-1)
    else:
        raise ValueError(f"Unsupported Dream visualization modality={modality!r}.")
    return {"image": primary, "wrist_image": wrist}


def load_episode_records(raw_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(Path(raw_dir).glob("replan_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No replan_*.pt prediction records found in {raw_dir}.")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]


def _sample_rows(arrays: list[np.ndarray], max_samples: int, seed: int = 0) -> np.ndarray:
    flat = np.concatenate([array.reshape(-1, array.shape[-1]) for array in arrays], axis=0)
    if len(flat) <= max_samples:
        return flat
    rng = np.random.default_rng(seed)
    return flat[rng.choice(len(flat), size=max_samples, replace=False)]


def fit_episode_projections(
    records: list[dict[str, Any]],
    *,
    sam_clusters: int,
    max_projection_samples: int,
) -> dict[str, Any]:
    """Fit stable colors jointly over cameras, horizons, and replans."""
    modality_arrays: dict[str, list[np.ndarray]] = {"dino": [], "sam": []}
    depth_values = []
    for record in records:
        predictions = record["dream_predictions"]
        for modality in ("dino", "sam"):
            if modality not in predictions:
                continue
            values = _as_numpy(predictions[modality])
            for horizon_prediction in values:
                views = split_prediction_views(modality, horizon_prediction)
                modality_arrays[modality].extend(views.values())
        if "depth" in predictions:
            for horizon_prediction in _as_numpy(predictions["depth"]):
                for depth in split_prediction_views("depth", horizon_prediction).values():
                    valid = depth[np.isfinite(depth) & (depth > 0)]
                    if valid.size:
                        depth_values.append(valid)

    projections: dict[str, Any] = {}
    if modality_arrays["dino"]:
        samples = _sample_rows(
            modality_arrays["dino"], max_samples=max_projection_samples, seed=0
        )
        projections["dino"] = fit_pca_projection(samples)
    if modality_arrays["sam"]:
        samples = _sample_rows(
            modality_arrays["sam"], max_samples=max_projection_samples, seed=1
        )
        projections["sam"] = fit_kmeans_projection(
            samples, clusters=sam_clusters
        )
    if depth_values:
        all_depth = np.concatenate(depth_values)
        projections["depth_range"] = tuple(
            float(x) for x in np.percentile(all_depth, [2.0, 98.0])
        )
    return projections


def _add_label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(output, (2, 2), (8 + width, 10 + height), (0, 0, 0), -1)
    cv2.putText(
        output, text, (5, 7 + height), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )
    return output


def _resize_panel(image: np.ndarray, panel_size: int) -> np.ndarray:
    return cv2.resize(_uint8_rgb(image), (panel_size, panel_size), interpolation=cv2.INTER_LINEAR)


def _render_prediction(
    modality: str,
    value: np.ndarray,
    rgb: np.ndarray,
    *,
    projections: dict[str, Any],
    alpha: float,
    sam_clusters: int,
    draw_contours: bool,
) -> np.ndarray:
    if modality == "depth":
        colored = depth_colormap(
            value,
            value_range=projections.get("depth_range"),
        )
        return colored
    if modality == "dyn":
        probability = 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))
        return overlay_dynamic_heatmap(rgb, probability, alpha=alpha, scale=1.0)
    if modality == "dino":
        return overlay_feature_pca(
            rgb,
            value,
            value.shape[:2],
            projection=projections.get("dino"),
            alpha=alpha,
        )
    if modality == "sam":
        return overlay_embedding_regions(
            rgb,
            value,
            value.shape[:2],
            clusters=sam_clusters,
            projection=projections.get("sam"),
            alpha=alpha,
            draw_contours=draw_contours,
        )
    raise ValueError(f"Unsupported modality={modality!r}.")


def render_record_grid(
    record: dict[str, Any],
    *,
    projections: dict[str, Any],
    panel_size: int,
    alpha: float,
    sam_clusters: int,
    draw_contours: bool,
) -> np.ndarray:
    rgb_by_camera = {
        camera: _uint8_rgb(record["rgb"][camera]) for camera in CAMERAS
    }
    predictions = record["dream_predictions"]
    future_offsets = [int(x) for x in record["future_offsets"]]
    modalities = [name for name in ("depth", "dyn", "dino", "sam") if name in predictions]
    rows: list[list[np.ndarray]] = [
        [
            _add_label(
                _resize_panel(rgb_by_camera[camera], panel_size),
                f"{camera} current RGB",
            )
            for camera in CAMERAS
        ]
    ]

    for modality in modalities:
        values = _as_numpy(predictions[modality])
        if values.shape[0] != len(future_offsets):
            raise ValueError(
                f"{modality} horizon count {values.shape[0]} does not match "
                f"future_offsets={future_offsets}."
            )
        for horizon_index, offset in enumerate(future_offsets):
            split = split_prediction_views(modality, values[horizon_index])
            row = []
            for camera in CAMERAS:
                rendered = _render_prediction(
                    modality,
                    split[camera],
                    rgb_by_camera[camera],
                    projections=projections,
                    alpha=alpha,
                    sam_clusters=sam_clusters,
                    draw_contours=draw_contours,
                )
                row.append(
                    _add_label(
                        _resize_panel(rendered, panel_size),
                        f"{camera} predicted {modality} t+{offset}",
                    )
                )
            rows.append(row)

    border = 2
    header_height = 28
    canvas_h = header_height + len(rows) * panel_size + (len(rows) + 1) * border
    canvas_w = len(CAMERAS) * panel_size + (len(CAMERAS) + 1) * border
    canvas = np.full((canvas_h, canvas_w, 3), 28, dtype=np.uint8)
    metadata = record.get("metadata", {})
    title = (
        f"replan={metadata.get('replan_index', '?')} "
        f"env_step={metadata.get('env_step', '?')} "
        f"success={metadata.get('success', False)}"
    )
    cv2.putText(
        canvas,
        title,
        (5, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    for row_index, row in enumerate(rows):
        y0 = header_height + border + row_index * (panel_size + border)
        for column_index, panel in enumerate(row):
            x0 = border + column_index * (panel_size + border)
            canvas[y0 : y0 + panel_size, x0 : x0 + panel_size] = panel
    return canvas


def render_saved_episode(
    raw_dir: Path,
    output_dir: Path,
    *,
    fps: int = 5,
    panel_size: int = 224,
    alpha: float = 0.5,
    sam_clusters: int = 8,
    draw_contours: bool = True,
    max_projection_samples: int = 2048,
) -> dict[str, str]:
    records = load_episode_records(raw_dir)
    projections = fit_episode_projections(
        records,
        sam_clusters=sam_clusters,
        max_projection_samples=max_projection_samples,
    )
    output_dir = Path(output_dir)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "dream_predictions.mp4"

    first_frame = render_record_grid(
        records[0],
        projections=projections,
        panel_size=panel_size,
        alpha=alpha,
        sam_clusters=sam_clusters,
        draw_contours=draw_contours,
    )
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (first_frame.shape[1], first_frame.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create visualization video: {video_path}")

    for index, record in enumerate(records):
        frame = first_frame if index == 0 else render_record_grid(
            record,
            projections=projections,
            panel_size=panel_size,
            alpha=alpha,
            sam_clusters=sam_clusters,
            draw_contours=draw_contours,
        )
        frame_path = frame_dir / f"replan_{index:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    result = {
        "video": str(video_path),
        "frames": str(frame_dir),
        "num_replans": str(len(records)),
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
