import hashlib
import os
from pathlib import Path
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


class DreamTargetAdapter:
    """Loads fixed-offset dense dream targets with LIBERO 2-camera horizontal concat."""

    MODALITIES = {"dyn", "depth", "dino", "sam"}
    DEFAULT_EXTRA_ROOTS = {
        "dyn": "cotracker",
        "depth": "depth_anything_v3_metric",
        "dino": "dinov2",
        "sam": "sam",
    }
    DEFAULT_ARRAY_KEYS = {
        "depth": "depth",
        "dino": "features",
        "sam": "features",
    }
    DEFAULT_FEATURE_DIMS = {
        "dino": 768,
        "sam": 256,
    }

    def __init__(self, dataset_dirs, cfg):
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = {} if cfg is None else dict(cfg)
        self.enabled = bool(cfg.get("enabled", False))
        self.mode = str(cfg.get("mode", "fixed_offset"))
        self.future_offset = int(cfg.get("future_offset", 4))
        self.modalities = list(cfg.get("modalities", ["dyn", "depth", "dino", "sam"]))
        self.modality_configs = {name: dict(cfg.get(name, {}) or {}) for name in self.MODALITIES}
        cameras = cfg.get("cameras", cfg.get("camera", ["image", "wrist_image"]))
        if isinstance(cameras, str):
            cameras = [cameras]
        self.cameras = list(cameras)
        self.array_keys = dict(self.DEFAULT_ARRAY_KEYS)
        self.array_keys.update(dict(cfg.get("array_keys", {}) or {}))
        self.extra_roots = dict(self.DEFAULT_EXTRA_ROOTS)
        self.extra_roots.update(dict(cfg.get("extra_roots", {}) or {}))
        self.feature_dims = dict(self.DEFAULT_FEATURE_DIMS)
        self.feature_dims.update(dict(cfg.get("feature_dims", {}) or {}))
        self.token_grids = dict(cfg.get("token_grids", {}) or {})
        self.target_shapes = {
            name: tuple(int(x) for x in shape)
            for name, shape in dict(cfg.get("target_shapes", {}) or {}).items()
            if shape is not None
        }
        self.dataset_dirs = [Path(p) for p in dataset_dirs]

        if self.enabled:
            if self.mode != "fixed_offset":
                raise ValueError(f"Unsupported dream_target.mode={self.mode!r}; only 'fixed_offset' is implemented.")
            if self.future_offset <= 0:
                raise ValueError(f"dream_target.future_offset must be > 0, got {self.future_offset}")
            unknown = set(self.modalities) - self.MODALITIES
            if unknown:
                raise ValueError(f"Unsupported dream_target.modalities: {sorted(unknown)}")
            if "depth" in self.modalities:
                depth_cfg = self.modality_configs.get("depth", {})
                if depth_cfg.get("source") != "depth_anything" or not depth_cfg.get("root"):
                    raise ValueError(
                        "dream_target depth supervision must use Depth Anything: set "
                        "dream_target.depth.source=depth_anything and dream_target.depth.root."
                    )
            if len(self.cameras) != 2:
                raise ValueError(
                    f"Dream target horizontal concat expects exactly 2 cameras, got {self.cameras}."
                )

    def has_enough_future(self, sample) -> bool:
        if not self.enabled:
            return True
        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is None:
            raise ValueError("dream_target requires `image_is_pad` to skip samples without t+future_offset.")
        if self.future_offset >= int(image_is_pad.shape[0]):
            return False
        return not bool(image_is_pad[self.future_offset].item())

    def build(self, sample):
        if not self.enabled:
            return None
        if "dataset_index" not in sample or "episode_index" not in sample or "frame_index" not in sample:
            raise ValueError(
                "dream_target requires dataset_index / episode_index / frame_index, "
                "but they were not found after preprocessing"
            )
        dataset_index = int(torch.as_tensor(sample["dataset_index"]).item())
        episode_index = int(torch.as_tensor(sample["episode_index"]).item())
        frame_index = int(torch.as_tensor(sample["frame_index"]).item())
        target_frame = frame_index + self.future_offset
        ds_root = self.dataset_dirs[dataset_index]
        targets = {}
        for modality in self.modalities:
            if modality == "dyn":
                targets[modality] = self._load_dyn(ds_root, episode_index, frame_index, target_frame)
            else:
                targets[modality] = self._load_single_frame(ds_root, modality, episode_index, target_frame)
            self._validate_target_shape(modality, targets[modality])
        return targets

    def _npz_path(self, ds_root: Path, modality: str, camera: str, episode_index: int) -> Path:
        modality_cfg = self.modality_configs.get(modality, {})
        root = modality_cfg.get("root")
        if root:
            root = Path(root)
            candidates = [
                root / ds_root.name / camera / f"episode_{episode_index:06d}.npz",
                root / camera / f"episode_{episode_index:06d}.npz",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return candidates[0]
        extra_root = self.extra_roots.get(modality, modality)
        return ds_root / "extras" / extra_root / camera / f"episode_{episode_index:06d}.npz"

    def _frame_row(self, payload, path: Path, frame_index: int) -> int:
        if "frame_index" not in payload.files:
            return frame_index
        frame_indices = payload["frame_index"]
        matches = np.nonzero(frame_indices == frame_index)[0]
        if len(matches) != 1:
            raise IndexError(
                f"Expected exactly one frame_index={frame_index} in {path}, found {len(matches)}."
            )
        return int(matches[0])

    def _load_npz_array(self, path: Path, modality: str, frame_index: int) -> torch.Tensor:
        if not path.exists():
            if modality == "depth" and self.modality_configs.get("depth", {}).get("root"):
                raise FileNotFoundError(
                    f"Missing Depth Anything dream target: {path}. "
                    "Generate Depth Anything depth first; depth dream supervision does not fallback to the raw depth folder."
                )
            raise FileNotFoundError(
                f"Missing dream target data for modality={modality!r}: {path}. "
                "Run the corresponding preprocessing job or disable/remove this modality in dream_target.modalities."
            )
        with np.load(path) as payload:
            key = self.array_keys.get(modality, modality)
            if key not in payload.files:
                raise KeyError(f"{path} does not contain key {key!r}; available keys={payload.files}.")
            arr = payload[key]
            row = self._frame_row(payload, path, frame_index)
            if row >= arr.shape[0]:
                raise IndexError(
                    f"Dream target row {row} for frame {frame_index} out of bounds for {path}: first dimension={arr.shape[0]}"
                )
            if "valid" in payload.files and not bool(payload["valid"][row]):
                raise ValueError(f"Dream target frame {frame_index} is marked invalid in {path}.")
            return torch.as_tensor(arr[row]).float()

    def _load_single_frame(self, ds_root: Path, modality: str, episode_index: int, target_frame: int) -> torch.Tensor:
        tensors = [
            self._load_npz_array(self._npz_path(ds_root, modality, camera, episode_index), modality, target_frame)
            for camera in self.cameras
        ]
        return self._concat_camera_targets(tensors[0], tensors[1], modality)

    def _load_dyn(self, ds_root: Path, episode_index: int, frame_index: int, target_frame: int) -> torch.Tensor:
        dyn_tensors = []
        for camera in self.cameras:
            path = self._npz_path(ds_root, "dyn", camera, episode_index)
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing CoTracker dynamic dream target data: {path}. "
                    "Expected extras/cotracker/{image,wrist_image}/episode_XXXXXX.npz with tracks/visibility."
                )
            with np.load(path) as payload:
                if "tracks" in payload.files:
                    tracks = payload["tracks"]
                    start_row = self._frame_row(payload, path, frame_index)
                    target_row = self._frame_row(payload, path, target_frame)
                    if "valid" in payload.files and (
                        not bool(payload["valid"][start_row]) or not bool(payload["valid"][target_row])
                    ):
                        raise ValueError(f"CoTracker frame pair {frame_index}->{target_frame} is invalid in {path}.")
                    if tracks.ndim != 3 or tracks.shape[-1] != 2:
                        raise ValueError(f"{path} key 'tracks' must have shape [T, N, 2], got {tracks.shape}.")
                    delta = torch.as_tensor(tracks[target_row]).float() - torch.as_tensor(tracks[start_row]).float()
                    motion = delta.norm(dim=-1, keepdim=True)
                    if "visibility" in payload.files:
                        vis = torch.as_tensor(payload["visibility"][target_row]).float().reshape(-1, 1)
                        motion = motion * vis
                    max_motion = motion.max().clamp(min=1e-6)
                    motion = (motion / max_motion).clamp(0.0, 1.0)
                    dyn_tensors.append(motion)
                elif "dyn" in payload.files:
                    arr = payload["dyn"]
                    if arr.ndim < 2 or target_frame >= arr.shape[1]:
                        raise ValueError(
                            f"{path} key 'dyn' must be indexable as [start_frame, end_frame, ...] "
                            f"for pair {frame_index}->{target_frame}, got shape {arr.shape}."
                        )
                    dyn_tensors.append(torch.as_tensor(arr[frame_index, target_frame]).float())
                else:
                    raise ValueError(
                        f"{path} must contain CoTracker keys 'tracks'/'visibility' or dense key 'dyn' "
                        "for t->t+future_offset labels."
                    )
        return self._concat_camera_targets(dyn_tensors[0], dyn_tensors[1], "dyn")

    def _validate_target_shape(self, modality: str, tensor: torch.Tensor):
        expected = self.target_shapes.get(modality)
        if expected is None:
            return
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"dream_targets[{modality!r}] shape mismatch in dataset adapter: "
                f"expected {expected}, got {tuple(tensor.shape)}"
            )

    def _concat_camera_targets(self, primary: torch.Tensor, wrist: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == "depth":
            return self.concat_2cam_images_horiz(primary, wrist, modality)
        if modality == "dyn":
            primary_mask = self._dynamic_score_to_grid(primary)
            wrist_mask = self._dynamic_score_to_grid(wrist)
            return torch.cat([primary_mask, wrist_mask], dim=1).contiguous()
        if modality in {"dino", "sam"}:
            primary_tokens = self._as_token_matrix(primary, modality)
            wrist_tokens = self._as_token_matrix(wrist, modality)
            grid_h, grid_w = self._infer_token_grid(modality, primary_tokens.shape[0])
            return self.concat_2cam_tokens_horiz(primary_tokens, wrist_tokens, grid_h, grid_w)
        raise ValueError(f"Unsupported dream modality={modality!r}")

    @staticmethod
    def concat_2cam_images_horiz(primary: torch.Tensor, wrist: torch.Tensor, modality: str) -> torch.Tensor:
        primary = primary.float()
        wrist = wrist.float()
        if primary.ndim == 3 and primary.shape[0] == 1:
            primary = primary.squeeze(0)
        if wrist.ndim == 3 and wrist.shape[0] == 1:
            wrist = wrist.squeeze(0)
        if primary.ndim != 2 or wrist.ndim != 2:
            raise ValueError(
                f"{modality} image-like targets must be [H,W] or [1,H,W], got "
                f"{tuple(primary.shape)} and {tuple(wrist.shape)}"
            )
        if primary.shape[0] != wrist.shape[0]:
            raise ValueError(
                f"{modality} camera target height mismatch: {tuple(primary.shape)} vs {tuple(wrist.shape)}"
            )
        return torch.cat([primary, wrist], dim=1).contiguous()

    @staticmethod
    def concat_2cam_tokens_horiz(primary: torch.Tensor, wrist: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        if primary.ndim != 2 or wrist.ndim != 2:
            raise ValueError(
                f"Token targets must be [N,C], got {tuple(primary.shape)} and {tuple(wrist.shape)}"
            )
        if primary.shape != wrist.shape:
            raise ValueError(f"Token camera target shape mismatch: {tuple(primary.shape)} vs {tuple(wrist.shape)}")
        if primary.shape[0] != grid_h * grid_w:
            raise ValueError(
                f"Token grid {grid_h}x{grid_w} does not match token count {primary.shape[0]}"
            )
        c = primary.shape[1]
        primary_grid = primary.reshape(grid_h, grid_w, c)
        wrist_grid = wrist.reshape(grid_h, grid_w, c)
        return torch.cat([primary_grid, wrist_grid], dim=1).contiguous()

    def _as_token_matrix(self, tensor: torch.Tensor, modality: str) -> torch.Tensor:
        tensor = tensor.float()
        feature_dim = int(self.feature_dims[modality])
        if tensor.ndim == 1:
            if tensor.numel() % feature_dim != 0:
                raise ValueError(
                    f"{modality} flat target length {tensor.numel()} is not divisible by feature_dim={feature_dim}."
                )
            return tensor.reshape(-1, feature_dim)
        if tensor.ndim == 2:
            if tensor.shape[-1] == feature_dim:
                return tensor
            if tensor.numel() % feature_dim == 0:
                return tensor.reshape(-1, feature_dim)
        if tensor.ndim >= 3 and tensor.shape[-1] == feature_dim:
            return tensor.reshape(-1, feature_dim)
        raise ValueError(
            f"Cannot convert {modality} target shape {tuple(tensor.shape)} to [N,{feature_dim}]."
        )

    def _dynamic_score_to_grid(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float()
        if tensor.ndim == 2 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
        if tensor.ndim == 1:
            grid_h, grid_w = self._infer_token_grid("dyn", tensor.numel())
            return tensor.reshape(grid_h, grid_w).clamp(0.0, 1.0)
        if tensor.ndim == 2:
            return tensor.clamp(0.0, 1.0)
        raise ValueError(f"Cannot convert dyn target shape {tuple(tensor.shape)} to dynamic mask grid.")

    def _infer_token_grid(self, modality: str, num_tokens: int) -> tuple[int, int]:
        configured = self.token_grids.get(modality)
        if configured is not None:
            grid_h, grid_w = int(configured[0]), int(configured[1])
            if grid_h * grid_w != int(num_tokens):
                raise ValueError(
                    f"dream_target.token_grids.{modality}={configured} does not match token count {num_tokens}."
                )
            return grid_h, grid_w
        side = int(round(float(num_tokens) ** 0.5))
        if side * side == int(num_tokens):
            return side, side
        raise ValueError(
            f"Cannot infer 2D grid for {modality} with {num_tokens} tokens. "
            f"Set dream_target.token_grids.{modality}: [grid_h, grid_w]."
        )

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        dataset_name: Optional[str] = None,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        dream_target=None,
    ):
        self.dataset_name = dataset_name
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.dream_target_adapter = DreamTargetAdapter(dataset_dirs=dataset_dirs, cfg=dream_target)

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def debug_sample_metadata(self, idx: int = 0):
        sample = self[idx]
        print("sample.keys()", sorted(sample.keys()))
        for key in ("dataset_index", "episode_index", "frame_index"):
            print(f"sample[{key!r}]", sample[key])
        if "dream_targets" in sample:
            print("dream_targets", {k: tuple(v.shape) for k, v in sample["dream_targets"].items()})
        return sample

    def discover_dream_target_shapes(self, idx: int = 0):
        if not self.dream_target_adapter.enabled:
            raise ValueError("dream_target.enabled must be true for shape discovery.")
        sample = self[idx]
        if "dream_targets" not in sample:
            raise ValueError("Sample does not contain dream_targets; check dream_target configuration.")
        shapes = {key: list(value.shape) for key, value in sample["dream_targets"].items()}
        print("dream_target_shapes", shapes)
        return shapes

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            needs_resample = False
            if self.dream_target_adapter.enabled and not self.dream_target_adapter.has_enough_future(sample):
                needs_resample = True

            if not self.skip_padding_as_possible and not needs_resample:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = needs_resample
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        dream_targets = self.dream_target_adapter.build(sample)
        if dream_targets is not None:
            data["dream_targets"] = dream_targets
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
