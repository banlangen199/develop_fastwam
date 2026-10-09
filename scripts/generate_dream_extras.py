"""Generate the dream-target extras that `RobotVideoDataset` expects.

Why this exists
---------------
`configs/data/*_2cam.yaml` asks for per-frame future targets under
`extras/<root>/<camera>/episode_XXXXXX.*`, but the LIBERO datasets published to
the bucket contain only `data/ meta/ videos/`. When the extras are absent the
dataset does not always raise: on several paths it zero-fills and marks the
future invalid, so training runs to completion with `loss_dream` pinned at
exactly 0.0000 and a dream branch that has learned nothing. That was observed on
an internal run. Regenerating the extras is the only fix.

Scope: `dino` (semantic) and `depth` (geometric). Those are the two the
stage-dependent-perception study needs -- `sam` would duplicate `dino`'s role
and `dyn` needs optical flow, so both are out of scope here and must stay out of
`dream_target.modalities` unless they are generated too.

Layout produced (matching `RobotVideoDataset._extra_path` and the `.npy`
sidecar branch of `_load_extra_array`)::

    extras/dinov2/{image,wrist_image}/episode_XXXXXX.features.npy   [T, 256, 768] fp16
    extras/depth_anything_v3_metric/{image,wrist_image}/episode_XXXXXX.depth.npy
                                                                    [T, 128, 128] fp16

Shape contract, derived from the loader rather than guessed:

* `dino` -- `feature_dims.dino=768`, `token_grids.dino=[16,16]`. Per camera the
  loader wants `[256, 768]` per frame; it reshapes to `(16,16,768)` and
  concatenates the two cameras along the width into the configured
  `target_shape=[16,32,768]`. DINOv2-base at 224x224 with patch 14 gives exactly
  16x16 patches, so no interpolation is needed.
* `depth` -- the loader concatenates the two cameras' depth *images*
  horizontally and only then patchifies with `patch_size=8`, so each camera must
  be 128x128: concat -> 128x256 -> 16x32=512 patches of 8x8=64, which is the
  configured `target_shape=[512,64]`.

    python scripts/generate_dream_extras.py --dataset <lerobot dir> \
        --modalities dino depth --device cuda:0
    # shard across GPUs:
    python scripts/generate_dream_extras.py --dataset <dir> --shard 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

CAMERA_KEYS = {"image": "observation.images.image", "wrist_image": "observation.images.wrist_image"}
EXTRA_ROOTS = {"dino": "dinov2", "depth": "depth_anything_v3_metric"}
ARRAY_KEYS = {"dino": "features", "depth": "depth"}

DINO_INPUT = 224          # 224 / patch 14 = 16 -> 16x16 = 256 tokens
DINO_GRID = 16
DINO_DIM = 768
DEPTH_SIDE = 128          # per camera; concat -> 128x256 -> /8 -> 16x32 = 512 patches


def read_video(path: Path) -> np.ndarray:
    """Decode an episode mp4 to uint8 [T, H, W, 3]."""
    import imageio.v3 as iio

    return np.asarray(iio.imread(path, plugin="pyav"))


def episode_lengths(dataset: Path) -> dict[int, int]:
    lengths = {}
    with (dataset / "meta" / "episodes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            lengths[int(record["episode_index"])] = int(record["length"])
    return lengths


def iter_episodes(dataset: Path, shard: int, num_shards: int) -> Iterator[tuple[int, int]]:
    for index, (episode, length) in enumerate(sorted(episode_lengths(dataset).items())):
        if index % num_shards == shard:
            yield episode, length


class DinoExtractor:
    def __init__(self, model_dir: str, device: str, dtype):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(device=device, dtype=dtype).eval()

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """uint8 [B,H,W,3] -> fp16 [B, 256, 768]."""
        torch = self.torch
        inputs = self.processor(images=list(frames), return_tensors="pt", size={"height": DINO_INPUT, "width": DINO_INPUT})
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            out = self.model(pixel_values=pixel_values).last_hidden_state
        # Drop the CLS token (and any register tokens) and keep the patch grid.
        patches = out[:, -DINO_GRID * DINO_GRID :, :]
        if patches.shape[1] != DINO_GRID * DINO_GRID or patches.shape[2] != DINO_DIM:
            raise RuntimeError(
                f"DINO produced {tuple(patches.shape[1:])}, expected "
                f"({DINO_GRID * DINO_GRID}, {DINO_DIM}). Check the checkpoint and input size."
            )
        return patches.float().cpu().numpy().astype(np.float16)


class DepthExtractor:
    def __init__(self, model_dir: str, device: str, dtype):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(model_dir).to(device=device, dtype=dtype).eval()
        )

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """uint8 [B,H,W,3] -> fp16 [B, 128, 128]."""
        torch = self.torch
        inputs = self.processor(images=list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            predicted = self.model(pixel_values=pixel_values).predicted_depth
        if predicted.ndim == 3:
            predicted = predicted.unsqueeze(1)
        resized = torch.nn.functional.interpolate(
            predicted.float(), size=(DEPTH_SIDE, DEPTH_SIDE), mode="bilinear", align_corners=False
        ).squeeze(1)
        return resized.cpu().numpy().astype(np.float16)


def sidecar_path(dataset: Path, modality: str, camera: str, episode: int) -> Path:
    return (
        dataset
        / "extras"
        / EXTRA_ROOTS[modality]
        / camera
        / f"episode_{episode:06d}.{ARRAY_KEYS[modality]}.npy"
    )


def process(args) -> int:
    import torch

    dataset = Path(args.dataset)
    if not (dataset / "meta" / "episodes.jsonl").is_file():
        print(f"not a lerobot dataset: {dataset}", file=sys.stderr)
        return 2

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    extractors = {}
    if "dino" in args.modalities:
        extractors["dino"] = DinoExtractor(args.dino_model, args.device, dtype)
    if "depth" in args.modalities:
        extractors["depth"] = DepthExtractor(args.depth_model, args.device, dtype)

    for modality in args.modalities:
        for camera in CAMERA_KEYS:
            sidecar_path(dataset, modality, camera, 0).parent.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    for episode, length in iter_episodes(dataset, args.shard, args.num_shards):
        targets = {
            (modality, camera): sidecar_path(dataset, modality, camera, episode)
            for modality in args.modalities
            for camera in CAMERA_KEYS
        }
        if not args.overwrite and all(p.is_file() for p in targets.values()):
            skipped += 1
            continue

        for camera, video_key in CAMERA_KEYS.items():
            video = dataset / "videos" / "chunk-000" / video_key / f"episode_{episode:06d}.mp4"
            if not video.is_file():
                print(f"[warn] missing video, skipping: {video}", file=sys.stderr)
                continue
            frames = read_video(video)
            if len(frames) != length:
                # The loader indexes rows by frame, so a length mismatch would
                # silently misalign every target with its observation.
                print(
                    f"[warn] episode {episode} camera {camera}: video has {len(frames)} frames, "
                    f"meta says {length}; using the video length",
                    file=sys.stderr,
                )

            for modality, extractor in extractors.items():
                out_path = targets[(modality, camera)]
                if out_path.is_file() and not args.overwrite:
                    continue
                chunks = []
                for start in range(0, len(frames), args.batch_size):
                    chunks.append(extractor(frames[start : start + args.batch_size]))
                array = np.concatenate(chunks, axis=0)
                # Stage locally then copy: the bucket is a FUSE mount that
                # rejects rename() with EPERM, so the usual write-tmp-then-
                # replace trick cannot be used on the destination itself.
                # Staging still keeps a crashed run from leaving a truncated
                # file that a later resume would skip as "already done";
                # --verify catches anything that slips through.
                import shutil
                import tempfile

                with tempfile.NamedTemporaryFile(
                    dir=args.scratch_dir, suffix=".npy", delete=False
                ) as handle:
                    np.save(handle, array)
                    staged = Path(handle.name)
                try:
                    shutil.copyfile(staged, out_path)
                finally:
                    staged.unlink(missing_ok=True)
        done += 1
        if done % 10 == 0:
            print(f"[extras] shard {args.shard}: {done} episodes written, {skipped} skipped")

    print(f"[extras] shard {args.shard} finished: {done} written, {skipped} skipped")
    return 0


def verify(args) -> int:
    """Check the produced arrays against the shapes the loader will demand."""
    dataset = Path(args.dataset)
    lengths = episode_lengths(dataset)
    expected = {
        "dino": (DINO_GRID * DINO_GRID, DINO_DIM),
        "depth": (DEPTH_SIDE, DEPTH_SIDE),
    }
    problems = 0
    checked = 0
    for episode, length in sorted(lengths.items()):
        for modality in args.modalities:
            for camera in CAMERA_KEYS:
                path = sidecar_path(dataset, modality, camera, episode)
                if not path.is_file():
                    print(f"[MISS] {path}")
                    problems += 1
                    continue
                array = np.load(path, mmap_mode="r")
                if array.shape[1:] != expected[modality]:
                    print(f"[SHAPE] {path}: {array.shape[1:]} != {expected[modality]}")
                    problems += 1
                elif array.shape[0] != length:
                    print(f"[LEN] {path}: {array.shape[0]} rows, meta says {length}")
                    problems += 1
                checked += 1
    print(f"\nchecked {checked} files, {problems} problem(s)")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--modalities", nargs="+", default=["dino", "depth"], choices=["dino", "depth"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--scratch-dir",
        default="/tmp",
        help="Local staging dir; must be on a filesystem that is NOT the bucket.",
    )
    parser.add_argument("--verify", action="store_true", help="Only check existing output.")
    parser.add_argument(
        "--dino-model",
        default=str(_REPO_ROOT.parent.parent / "_research" / "models" / "dinov2-base"),
    )
    parser.add_argument(
        "--depth-model",
        default=str(_REPO_ROOT.parent.parent / "_research" / "models" / "depth-anything-v2-small"),
    )
    args = parser.parse_args()
    return verify(args) if args.verify else process(args)


if __name__ == "__main__":
    raise SystemExit(main())
