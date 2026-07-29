"""Reusable visualization interfaces for SAM masks, dynamics and dense features.

Public functions accept and return RGB uint8 NumPy arrays. They do not read a
dataset, so ground-truth extras and model predictions can use the same API.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PCAProjection:
    """Reusable PCA transform; fit once to keep colors stable across frames."""

    mean: np.ndarray
    components: np.ndarray
    color_low: np.ndarray | None = None
    color_high: np.ndarray | None = None


@dataclass(frozen=True)
class KMeansProjection:
    """Reusable normalized cluster centers for temporally stable region colors."""

    centers: np.ndarray


def _rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must be [H,W,3], got {rgb.shape}")
    if rgb.dtype == np.uint8:
        return rgb
    rgb = rgb.astype(np.float32)
    if rgb.size and float(np.nanmax(rgb)) <= 1.5:
        rgb *= 255.0
    return np.nan_to_num(rgb).clip(0, 255).astype(np.uint8)


def _resize(array: np.ndarray, hw: tuple[int, int], nearest: bool = False) -> np.ndarray:
    h, w = hw
    mode = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(array, (w, h), interpolation=mode)


def depth_colormap(
    depth: np.ndarray,
    *,
    colormap: str = "inferno",
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Color a metric-depth map as an RGB uint8 image."""
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"depth must be [H,W], got {values.shape}")
    valid = np.isfinite(values) & (values > 0)
    if value_range is None:
        if np.any(valid):
            low, high = np.percentile(values[valid], [2.0, 98.0])
        else:
            low, high = 0.0, 1.0
    else:
        low, high = float(value_range[0]), float(value_range[1])
    high = max(float(high), float(low) + 1e-8)
    normalized = np.nan_to_num((values - low) / (high - low)).clip(0, 1)
    normalized_u8 = (normalized * 255).astype(np.uint8)
    cv_colormaps = {
        "inferno": cv2.COLORMAP_INFERNO,
        "hot": cv2.COLORMAP_HOT,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
    }
    if colormap not in cv_colormaps:
        raise ValueError(
            f"Unsupported depth colormap={colormap!r}; expected one of {sorted(cv_colormaps)}."
        )
    colored_bgr = cv2.applyColorMap(normalized_u8, cv_colormaps[colormap])
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    colored_rgb[~valid] = 0
    return colored_rgb


def overlay_depth(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    alpha: float = 0.55,
    colormap: str = "inferno",
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Color and blend a depth map with an RGB reference image."""
    image = _rgb_uint8(rgb)
    colored = depth_colormap(depth, colormap=colormap, value_range=value_range)
    if colored.shape[:2] != image.shape[:2]:
        colored = _resize(colored, image.shape[:2])
    return (
        image.astype(np.float32) * (1.0 - alpha)
        + colored.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)


def _palette(count: int, seed: int = 0) -> np.ndarray:
    colors = []
    for index in range(count):
        hue = ((seed * 0.137) + index * 0.618033988749895) % 1.0
        colors.append(colorsys.hsv_to_rgb(hue, 0.72, 1.0))
    return (np.asarray(colors) * 255).astype(np.uint8)


def overlay_sam_masks(
    rgb: np.ndarray,
    masks: np.ndarray,
    *,
    scores: np.ndarray | None = None,
    alpha: float = 0.48,
    draw_contours: bool = True,
    contour_width: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Overlay true/predicted masks.

    ``masks`` may be boolean ``[K,H,W]`` masks or an integer ``[H,W]`` region
    label map. This interface can directly consume a SAM decoder or model head.
    """
    image = _rgb_uint8(rgb)
    masks = np.asarray(masks)
    if masks.ndim == 2:
        mask_stack = np.stack([masks == label for label in np.unique(masks)])
    elif masks.ndim == 3:
        mask_stack = masks.astype(bool)
    else:
        raise ValueError(f"masks must be [K,H,W] or [H,W], got {masks.shape}")
    if mask_stack.shape[1:] != image.shape[:2]:
        mask_stack = np.stack([
            _resize(mask.astype(np.uint8), image.shape[:2], True).astype(bool)
            for mask in mask_stack
        ])
    order = np.arange(len(mask_stack))
    if scores is not None:
        scores = np.asarray(scores)
        if scores.shape != (len(mask_stack),):
            raise ValueError(f"scores must be {(len(mask_stack),)}, got {scores.shape}")
        order = np.argsort(scores)
    result = image.astype(np.float32)
    colors = _palette(len(mask_stack), seed)
    for index in order:
        mask = mask_stack[index]
        if not np.any(mask):
            continue
        color = colors[index].astype(np.float32)
        result[mask] = result[mask] * (1.0 - alpha) + color * alpha
        if draw_contours:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, color.tolist(), contour_width, cv2.LINE_AA)
    return result.clip(0, 255).astype(np.uint8)


def dynamic_map_from_tracks(
    tracks: np.ndarray,
    visibility: np.ndarray | None = None,
    *,
    grid_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert CoTracker displacement vectors [P,2]/[H,W,2] to motion magnitude."""
    tracks = np.asarray(tracks, dtype=np.float32)
    if tracks.ndim == 3 and tracks.shape[-1] == 2:
        grid_hw = grid_hw or tracks.shape[:2]
    elif tracks.ndim == 2 and tracks.shape[-1] == 2:
        if grid_hw is None:
            side = int(round(np.sqrt(len(tracks))))
            if side * side != len(tracks):
                raise ValueError("non-square track count requires grid_hw")
            grid_hw = (side, side)
    else:
        raise ValueError(f"tracks must be [P,2] or [H,W,2], got {tracks.shape}")
    magnitude = np.linalg.norm(tracks, axis=-1).reshape(grid_hw)
    if visibility is not None:
        magnitude *= np.clip(np.asarray(visibility).reshape(grid_hw), 0, 1)
    return magnitude


def dynamic_heatmap(
    dynamic: np.ndarray,
    *,
    output_hw: tuple[int, int] | None = None,
    percentile: float = 95.0,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return red-yellow RGB heatmap and normalized [0,1] motion strength."""
    values = np.asarray(dynamic, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"dynamic map must be [H,W], got {values.shape}")
    positive = values[np.isfinite(values) & (values > 0)]
    scale = scale if scale is not None else (
        float(np.percentile(positive, percentile)) if positive.size else 1.0
    )
    strength = np.nan_to_num(values / max(scale, 1e-8)).clip(0, 1)
    if output_hw is not None and strength.shape != output_hw:
        strength = _resize(strength, output_hw)
    heatmap = np.stack([
        np.clip(strength * 2, 0, 1),
        np.clip(strength * 2 - 1, 0, 1),
        np.zeros_like(strength),
    ], axis=-1)
    return (heatmap * 255).astype(np.uint8), strength


def overlay_dynamic_heatmap(
    rgb: np.ndarray,
    dynamic: np.ndarray,
    *,
    alpha: float = 0.72,
    percentile: float = 95.0,
    scale: float | None = None,
) -> np.ndarray:
    """Overlay motion heat; zero-motion pixels leave the source RGB unchanged."""
    image = _rgb_uint8(rgb)
    heatmap, strength = dynamic_heatmap(
        dynamic, output_hw=image.shape[:2], percentile=percentile, scale=scale
    )
    weight = (alpha * strength)[..., None]
    return (image * (1 - weight) + heatmap * weight).clip(0, 255).astype(np.uint8)


def as_dense_features(
    features: np.ndarray,
    spatial_hw: tuple[int, int],
    *,
    layout: str = "auto",
) -> np.ndarray:
    """Convert token, CHW, HWC or flattened feature storage to [H,W,C]."""
    features = np.asarray(features)
    h, w = spatial_hw
    pixels = h * w
    if features.ndim == 3:
        if layout == "chw" or (layout == "auto" and features.shape[1:] == (h, w)):
            return features.transpose(1, 2, 0).astype(np.float32)
        if features.shape[:2] == (h, w):
            return features.astype(np.float32)
    if features.ndim == 2:
        if features.shape[0] == pixels:
            return features.reshape(h, w, -1).astype(np.float32)
        if features.shape[1] == pixels:
            return features.reshape(-1, h, w).transpose(1, 2, 0).astype(np.float32)
    if features.ndim == 1 and features.size % pixels == 0:
        # Current SAM extras store image_encoder [C,H,W] via flatten(1).
        return features.reshape(-1, h, w).transpose(1, 2, 0).astype(np.float32)
    raise ValueError(f"cannot reshape {features.shape} to dense grid {spatial_hw}")


def fit_pca_projection(dense_features: np.ndarray, components: int = 3) -> PCAProjection:
    """Fit a NumPy PCA transform on [H,W,C] or [N,C] features."""
    dense = np.asarray(dense_features, dtype=np.float32)
    flat = dense.reshape(-1, dense.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    _, _, vh = np.linalg.svd(flat - mean, full_matrices=False)
    basis = vh[:components].astype(np.float32)
    projected = (flat - mean) @ basis.T
    color_low = np.percentile(projected, 2.0, axis=0).astype(np.float32)
    color_high = np.percentile(projected, 98.0, axis=0).astype(np.float32)
    return PCAProjection(mean.astype(np.float32), basis, color_low, color_high)


def feature_pca_rgb(
    features: np.ndarray,
    spatial_hw: tuple[int, int],
    *,
    projection: PCAProjection | None = None,
    output_hw: tuple[int, int] | None = None,
    percentile_range: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """PCA-project dense features to an RGB image."""
    dense = as_dense_features(features, spatial_hw)
    projection = projection or fit_pca_projection(dense)
    reduced = ((dense.reshape(-1, dense.shape[-1]) - projection.mean)
               @ projection.components.T).reshape(*spatial_hw, -1)
    if reduced.shape[-1] < 3:
        reduced = np.pad(reduced, ((0, 0), (0, 0), (0, 3 - reduced.shape[-1])))
    rgb = np.empty((*spatial_hw, 3), dtype=np.float32)
    for channel in range(3):
        if projection.color_low is not None and projection.color_high is not None:
            low = float(projection.color_low[channel])
            high = float(projection.color_high[channel])
        else:
            low, high = np.percentile(reduced[..., channel], percentile_range)
        rgb[..., channel] = np.clip(
            (reduced[..., channel] - low) / max(high - low, 1e-8), 0, 1
        )
    result = (rgb * 255).astype(np.uint8)
    return _resize(result, output_hw) if output_hw is not None else result


def overlay_feature_pca(
    rgb: np.ndarray,
    features: np.ndarray,
    spatial_hw: tuple[int, int],
    *,
    projection: PCAProjection | None = None,
    alpha: float = 0.48,
) -> np.ndarray:
    """PCA-color DINO/SAM/model dense features and blend with RGB."""
    image = _rgb_uint8(rgb)
    colors = feature_pca_rgb(features, spatial_hw, projection=projection, output_hw=image.shape[:2])
    return (image * (1 - alpha) + colors * alpha).clip(0, 255).astype(np.uint8)


def kmeans_feature_regions(
    features: np.ndarray,
    spatial_hw: tuple[int, int],
    *,
    clusters: int = 8,
    iterations: int = 30,
    seed: int = 0,
    projection: KMeansProjection | None = None,
) -> np.ndarray:
    """Cluster dense embeddings into a region label map without sklearn."""
    dense = as_dense_features(features, spatial_hw)
    flat = dense.reshape(-1, dense.shape[-1]).astype(np.float32)
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)
    if projection is not None:
        return np.argmax(flat @ projection.centers.T, axis=1).reshape(spatial_hw)
    if not 1 <= clusters <= len(flat):
        raise ValueError(f"clusters must be in [1,{len(flat)}]")
    rng = np.random.default_rng(seed)
    centers = flat[rng.choice(len(flat), clusters, replace=False)].copy()
    labels = np.full(len(flat), -1, dtype=np.int32)
    for _ in range(iterations):
        new_labels = np.argmax(flat @ centers.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(clusters):
            members = flat[labels == index]
            if len(members):
                centers[index] = members.mean(axis=0)
                centers[index] /= max(float(np.linalg.norm(centers[index])), 1e-8)
    return labels.reshape(spatial_hw)


def fit_kmeans_projection(
    dense_features: np.ndarray,
    *,
    clusters: int = 8,
    iterations: int = 40,
    seed: int = 0,
) -> KMeansProjection:
    """Fit reusable cosine-KMeans centers on [N,C] or [...,C] features."""
    flat = np.asarray(dense_features, dtype=np.float32)
    flat = flat.reshape(-1, flat.shape[-1])
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)
    if not 1 <= clusters <= len(flat):
        raise ValueError(f"clusters must be in [1,{len(flat)}]")
    rng = np.random.default_rng(seed)
    centers = flat[rng.choice(len(flat), clusters, replace=False)].copy()
    labels = np.full(len(flat), -1, dtype=np.int32)
    for _ in range(iterations):
        new_labels = np.argmax(flat @ centers.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(clusters):
            members = flat[labels == index]
            if len(members):
                centers[index] = members.mean(axis=0)
                centers[index] /= max(float(np.linalg.norm(centers[index])), 1e-8)
    return KMeansProjection(centers)


def overlay_embedding_regions(
    rgb: np.ndarray,
    features: np.ndarray,
    spatial_hw: tuple[int, int],
    *,
    clusters: int = 8,
    alpha: float = 0.45,
    draw_contours: bool = True,
    seed: int = 0,
    projection: KMeansProjection | None = None,
) -> np.ndarray:
    """Approximate region overlay for embeddings when true masks are unavailable."""
    image = _rgb_uint8(rgb)
    labels = kmeans_feature_regions(
        features, spatial_hw, clusters=clusters, seed=seed, projection=projection
    )
    labels = _resize(labels.astype(np.int32), image.shape[:2], True)
    return overlay_sam_masks(
        image, labels, alpha=alpha, draw_contours=draw_contours, seed=seed
    )
