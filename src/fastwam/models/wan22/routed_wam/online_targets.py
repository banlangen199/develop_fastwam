"""Compute dream targets online, from frames the batch already contains.

Why online rather than precomputed extras
-----------------------------------------
`configs/data/libero_2cam.yaml` samples a 33-frame clip and keeps every 4th
frame, so ``video_sample_indices == [0, 4, 8, ..., 32]``. The configured
``future_offsets`` of 16 and 32 are therefore video-tensor indices 4 and 8 --
**the future frames are already in the batch**. Extracting their targets on the
GPU costs two forwards of small frozen networks over tensors that are already
resident, and needs no extra decoding and no extra I/O at all.

Precomputing the same targets to `extras/` costs, for one LIBERO suite, ~44 GB
that then has to be read back over a FUSE-mounted bucket at roughly 48 MB per
training step (DINO alone is 1.5 MB per sample at two offsets and two cameras).
That read is plausibly slower than recomputation, and it has to be regenerated
whenever `future_offsets` or `modalities` change.

All preprocessing here is done as tensor ops on device. The HuggingFace image
processors are deliberately not used at train time: they are PIL-based, run on
one CPU thread, and measured at 327 ms (DINOv2) and 729 ms (Depth-Anything) for
a single 128-image step -- overwhelmingly preprocessing, not model compute.

Layout produced, matching what `DreamFastWAM._compute_dream_loss` and the
`DenseDreamDecoder` target shapes expect:

    dino  -> [B, O, 16, 32, 768]   two 16x16 patch grids concatenated by width
    depth -> [B, O, 512, 64]       two 128x128 maps concatenated to 128x256,
                                   then patchified with patch_size 8

The two cameras arrive already concatenated along width (``video_size``
``[224, 448]``, ``concat_multi_camera: horizontal``), so the split is a slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger


logger = get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINO_GRID = 16
DINO_DIM = 768
DEPTH_SIDE = 128
DEPTH_PATCH = 8


@dataclass(frozen=True)
class OnlineTargetConfig:
    enabled: bool = False
    modalities: tuple[str, ...] = ("dino", "depth")
    dino_model: Optional[str] = None
    depth_model: Optional[str] = None
    dino_input: int = 224
    depth_input: int = 224
    #: Depth-Anything predicts *relative* depth: each map is defined only up to
    #: an affine transform, so the raw scale drifts frame to frame. Supervising
    #: against it makes the model spend capacity on a scale it cannot infer from
    #: the input. `robust` removes that nuisance degree of freedom by centring on
    #: the median and dividing by the 10-90 percentile spread, per camera --
    #: per camera rather than over the concatenated pair because the wrist view
    #: is a close-up with a very different range, and the decoders reconstruct
    #: each camera from its own tokens anyway.
    depth_normalize: str = "robust"
    #: Pixel range of `sample["video"]`. The VAE path feeds [-1, 1]; the
    #: extractors need [0, 1], so the conversion has to know which it is.
    video_range: str = "tanh"
    dtype: str = "float16"

    @classmethod
    def from_dict(cls, value: Optional[dict[str, Any]]) -> "OnlineTargetConfig":
        payload = dict(value or {})
        if "modalities" in payload:
            payload["modalities"] = tuple(payload["modalities"])
        cfg = cls(**payload)
        unknown = set(cfg.modalities) - {"dino", "depth"}
        if unknown:
            raise ValueError(
                f"online_dream_targets currently supports dino and depth, got {sorted(unknown)}. "
                "sam and dyn need their own extractors before they can be listed."
            )
        if cfg.depth_normalize not in ("none", "robust"):
            raise ValueError(
                f"depth_normalize must be 'none' or 'robust', got {cfg.depth_normalize!r}."
            )
        if cfg.video_range not in ("tanh", "unit"):
            raise ValueError(f"video_range must be 'tanh' or 'unit', got {cfg.video_range!r}.")
        if cfg.enabled and "dino" in cfg.modalities and not cfg.dino_model:
            raise ValueError("online_dream_targets.dino_model must be set when dino is enabled.")
        if cfg.enabled and "depth" in cfg.modalities and not cfg.depth_model:
            raise ValueError("online_dream_targets.depth_model must be set when depth is enabled.")
        return cfg


def patchify_depth(image: torch.Tensor, patch_size: int = DEPTH_PATCH) -> torch.Tensor:
    """[B, H, W] -> [B, (H/p)*(W/p), p*p], row-major over patches.

    Mirrors ``DreamTargetAdapter._patchify_image_target``;
    ``tests/test_online_targets.py`` asserts the two agree element-wise, so this
    copy cannot drift from the loader it has to match.
    """
    if image.ndim != 3:
        raise ValueError(f"expected [B,H,W], got {tuple(image.shape)}")
    batch, height, width = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"{height}x{width} is not divisible by patch_size={patch_size}")
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = image.reshape(batch, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(0, 1, 3, 2, 4).contiguous()
    return patches.reshape(batch, grid_h * grid_w, patch_size * patch_size)


class OnlineDreamTargets(nn.Module):
    """Frozen extractors that turn future frames into dream targets on device."""

    def __init__(self, config: OnlineTargetConfig):
        super().__init__()
        self.config = config
        self.dino = None
        self.depth = None
        if not config.enabled:
            return

        from transformers import AutoModel, AutoModelForDepthEstimation

        if "dino" in config.modalities:
            self.dino = AutoModel.from_pretrained(config.dino_model).eval()
            self.dino.requires_grad_(False)
        if "depth" in config.modalities:
            self.depth = AutoModelForDepthEstimation.from_pretrained(config.depth_model).eval()
            self.depth.requires_grad_(False)

        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        logger.info(
            "OnlineDreamTargets enabled for %s (dino=%s, depth=%s)",
            list(config.modalities),
            config.dino_model,
            config.depth_model,
        )

    # ------------------------------------------------------------------ utils
    def _to_unit(self, frames: torch.Tensor) -> torch.Tensor:
        """Whatever range the video tensor uses -> [0, 1]."""
        if self.config.video_range == "tanh":
            frames = (frames + 1.0) * 0.5
        return frames.clamp(0.0, 1.0)

    def _normalise(self, frames: torch.Tensor, size: int) -> torch.Tensor:
        frames = F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)
        return (frames - self._mean.to(frames.dtype)) / self._std.to(frames.dtype)

    @staticmethod
    def split_cameras(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B,3,H,2W] -> primary [B,3,H,W], wrist [B,3,H,W]."""
        width = frames.shape[-1]
        if width % 2:
            raise ValueError(f"two-camera frame width must be even, got {width}")
        half = width // 2
        return frames[..., :half], frames[..., half:]

    # ------------------------------------------------------------- extractors
    def _dino_grid(self, frames: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] in [0,1] -> [B, 16, 16, 768]."""
        pixel_values = self._normalise(frames, self.config.dino_input)
        out = self.dino(pixel_values=pixel_values).last_hidden_state
        tokens = out[:, -DINO_GRID * DINO_GRID :, :]
        if tokens.shape[1] != DINO_GRID * DINO_GRID or tokens.shape[2] != DINO_DIM:
            raise RuntimeError(
                f"DINO produced {tuple(tokens.shape[1:])}, expected "
                f"({DINO_GRID * DINO_GRID}, {DINO_DIM}); check dino_input vs the patch size."
            )
        return tokens.reshape(tokens.shape[0], DINO_GRID, DINO_GRID, DINO_DIM)

    @staticmethod
    def normalise_depth(depth: torch.Tensor, mode: str) -> torch.Tensor:
        """[B,H,W] relative depth -> scale-free [B,H,W].

        Percentiles rather than min/max: a single speckle pixel would otherwise
        set the scale for the whole map.
        """
        if mode == "none":
            return depth
        flat = depth.flatten(1)
        median = flat.median(dim=1, keepdim=True).values
        low = torch.quantile(flat, 0.1, dim=1, keepdim=True)
        high = torch.quantile(flat, 0.9, dim=1, keepdim=True)
        spread = (high - low).clamp(min=1e-6)
        return ((flat - median) / spread).reshape_as(depth)

    def _depth_map(self, frames: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] in [0,1] -> [B, 128, 128]."""
        pixel_values = self._normalise(frames, self.config.depth_input)
        predicted = self.depth(pixel_values=pixel_values).predicted_depth
        if predicted.ndim == 3:
            predicted = predicted.unsqueeze(1)
        resized = F.interpolate(
            predicted.float(), size=(DEPTH_SIDE, DEPTH_SIDE), mode="bilinear", align_corners=False
        ).squeeze(1)
        return self.normalise_depth(resized, self.config.depth_normalize)

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(
        self,
        video: torch.Tensor,
        *,
        future_offsets: list[int],
        action_video_freq_ratio: int,
        image_is_pad: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Build dream targets for every configured future offset.

        Args:
            video: ``[B, 3, T, H, 2W]`` -- the clip the video branch consumes,
                with the two cameras already concatenated along width.
            future_offsets: raw-frame offsets, e.g. ``[16, 32]``.
            action_video_freq_ratio: frames kept per video frame (4 for LIBERO),
                used to map a raw offset onto a video index.
            image_is_pad: ``[B, T_raw]`` boolean; a future that lands on padding
                is marked invalid instead of supervising against a repeated frame.

        Returns:
            ``{modality: [B, O, ...], f"{modality}_valid_mask": [B, O],
              "future_valid_mask": [B, O]}``
        """
        if not self.config.enabled:
            raise RuntimeError("OnlineDreamTargets.forward called while disabled.")
        if video.ndim != 5:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")

        batch, _, num_video_frames, _, _ = video.shape
        device = video.device
        targets: dict[str, list[torch.Tensor]] = {m: [] for m in self.config.modalities}
        valid: list[torch.Tensor] = []

        for offset in future_offsets:
            if offset % action_video_freq_ratio:
                raise ValueError(
                    f"future offset {offset} is not a multiple of "
                    f"action_video_freq_ratio={action_video_freq_ratio}, so it does not "
                    "land on a sampled video frame. Precomputed extras would be needed."
                )
            index = offset // action_video_freq_ratio
            if index >= num_video_frames:
                raise ValueError(
                    f"future offset {offset} maps to video index {index}, but the clip has "
                    f"only {num_video_frames} frames. Increase data.train.num_frames."
                )

            frames = self._to_unit(video[:, :, index].float())
            primary, wrist = self.split_cameras(frames)

            if "dino" in self.config.modalities:
                grid = torch.cat([self._dino_grid(primary), self._dino_grid(wrist)], dim=2)
                targets["dino"].append(grid)
            if "depth" in self.config.modalities:
                maps = torch.cat([self._depth_map(primary), self._depth_map(wrist)], dim=2)
                targets["depth"].append(patchify_depth(maps))

            if image_is_pad is None:
                valid.append(torch.ones((batch,), dtype=torch.bool, device=device))
            else:
                # `image_is_pad` is indexed by raw frame in the dataset's own
                # adapter, but the video tensor is subsampled. Detect which
                # convention this batch uses from its width rather than assuming:
                # guessing wrong marks every future invalid, the masked mean
                # collapses to 0/1, and training proceeds with `loss_dream`
                # pinned at exactly 0.0000 while looking entirely healthy.
                pad_len = int(image_is_pad.shape[1])
                if pad_len == num_video_frames:
                    column = index
                elif offset < pad_len:
                    column = offset
                else:
                    raise ValueError(
                        f"cannot validate future offset {offset}: image_is_pad has "
                        f"{pad_len} columns, which matches neither the raw-frame "
                        f"convention (needs > {offset}) nor the video-frame one "
                        f"(needs {num_video_frames}). Refusing to silently mark "
                        "every future invalid."
                    )
                valid.append(~image_is_pad[:, column].to(device=device, dtype=torch.bool))

        out: dict[str, Any] = {
            modality: torch.stack(values, dim=1) for modality, values in targets.items()
        }
        future_valid = torch.stack(valid, dim=1)
        out["future_valid_mask"] = future_valid
        for modality in self.config.modalities:
            out[f"{modality}_valid_mask"] = future_valid
        return out
