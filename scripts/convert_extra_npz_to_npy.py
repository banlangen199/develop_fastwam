#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import uuid

import numpy as np


DEFAULT_DATASET_DIRS = [
    "./data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot",
    "./data/libero_mujoco3.3.2/libero_object_no_noops_lerobot",
    "./data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot",
    "./data/libero_mujoco3.3.2/libero_10_no_noops_lerobot",
]

DEFAULT_EXTRA_ROOTS = [
    "cotracker",
    "depth_anything_v3_metric",
    "dinov2",
    "sam",
]

DEFAULT_CAMERAS = ["image", "wrist_image"]


def sidecar_path(npz_path: Path, key: str) -> Path:
    return npz_path.with_suffix(f".{key}.npy")


def save_sidecar(dst: Path, array: np.ndarray, *, overwrite: bool, dry_run: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    if dry_run:
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}.npy")
    try:
        np.save(tmp, array, allow_pickle=False)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    return True


def convert_npz(path: Path, *, keys: set[str] | None, overwrite: bool, dry_run: bool) -> tuple[int, int]:
    written = 0
    skipped = 0
    with np.load(path, allow_pickle=False) as payload:
        payload_keys = list(payload.files)
        if keys is not None:
            payload_keys = [key for key in payload_keys if key in keys]
        for key in payload_keys:
            dst = sidecar_path(path, key)
            if dst.exists() and not overwrite:
                skipped += 1
                continue
            if save_sidecar(dst, payload[key], overwrite=overwrite, dry_run=dry_run):
                written += 1
    return written, skipped


def iter_npz_files(dataset_dirs: list[Path], extra_roots: list[str], cameras: list[str]):
    for dataset_dir in dataset_dirs:
        for extra_root in extra_roots:
            for camera in cameras:
                root = dataset_dir / "extras" / extra_root / camera
                if not root.exists():
                    continue
                yield from sorted(root.glob("episode_*.npz"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create .npy sidecars for LIBERO dream-target extras. Original .npz files are not deleted. "
            "Each key in episode_XXXXXX.npz becomes episode_XXXXXX.<key>.npy."
        )
    )
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        default=DEFAULT_DATASET_DIRS,
        help="LeRobot dataset directories whose extras should be converted.",
    )
    parser.add_argument(
        "--extra-roots",
        nargs="+",
        default=DEFAULT_EXTRA_ROOTS,
        help="extras/<root>/<camera>/episode_*.npz roots to convert.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=DEFAULT_CAMERAS,
        help="Camera folders under each extra root.",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Optional subset of npz keys to export. By default all keys are exported.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing .npy sidecars.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be converted without writing files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of .npz files to process.")
    parser.add_argument("--log-every", type=int, default=100, help="Progress logging interval in .npz files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dirs = [Path(p) for p in args.dataset_dirs]
    keys = None if args.keys is None else set(args.keys)

    total_files = 0
    total_written = 0
    total_skipped = 0
    for path in iter_npz_files(dataset_dirs, args.extra_roots, args.cameras):
        if args.limit is not None and total_files >= args.limit:
            break
        written, skipped = convert_npz(path, keys=keys, overwrite=args.overwrite, dry_run=args.dry_run)
        total_files += 1
        total_written += written
        total_skipped += skipped
        if total_files == 1 or total_files % max(args.log_every, 1) == 0:
            action = "would write" if args.dry_run else "wrote"
            print(
                f"[{total_files}] {path} {action}={written} skipped={skipped} "
                f"total_written={total_written} total_skipped={total_skipped}",
                flush=True,
            )

    action = "would_write" if args.dry_run else "written"
    print(
        f"done files={total_files} {action}={total_written} skipped={total_skipped} "
        "original_npz_deleted=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
