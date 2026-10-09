#!/usr/bin/env python3
"""Run the trained RoutedWAM on held-out clips and dump everything a figure needs.

Produces one `.pt` payload per sample so the plotting pass can be re-run without
touching a GPU or re-reading a 10.9 GB checkpoint.

What is recorded, per sample:

* the RGB clip frames at t0 and at the two future offsets the Dream expert is
  asked about (16 and 32 raw frames ahead);
* the Dream expert's **prediction** for those futures (depth patches and DINO
  grids) alongside the **ground truth** the online extractors produce from the
  very same clip -- so the two are directly comparable, not merely plausible;
* the router's gate / keep decision for all 72 Dream tokens, at all 30 layers,
  at **every** action denoising step (the gate is a function of the action
  queries, so it moves as the action is denoised);
* where the action queries actually spend their attention: the per-dream-token
  attention they receive and the video / dream / action split of the total mass.
  A kept token can still be ignored, which the router's own keep_ratio cannot
  show;
* the denoised action chunk and the ground-truth action chunk.

Usage:
    PYTHONPATH=src python experiments/analysis/dump_dream_visuals.py \
        --run-dir runs/routed_wam_libero_4suite_full/<run_id> \
        --step 12500 --num-samples 4 --out evaluate_results/dream_visuals
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--step", type=int, default=12500)
    ap.add_argument("--ckpt", type=Path, default=None, help="overrides --step")
    ap.add_argument("--out", type=Path, default=Path("evaluate_results/dream_visuals"))
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--action-steps", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="explicit dataset indices; default is an even spread over the four suites",
    )
    return ap.parse_args()


def unpatchify_depth(patches: torch.Tensor, patch: int = 8, grid=(16, 32)) -> torch.Tensor:
    """[..., 512, 64] -> [..., 128, 256]; inverse of online_targets.patchify_depth."""
    grid_h, grid_w = grid
    lead = patches.shape[:-2]
    out = patches.reshape(*lead, grid_h, grid_w, patch, patch)
    out = out.permute(*range(len(lead)), -4, -2, -3, -1)
    return out.reshape(*lead, grid_h * patch, grid_w * patch)


def patch_datasets_column_access() -> None:
    """Make `hf_dataset[column]` return a list again.

    The vendored LeRobot loader does ``torch.stack(self.hf_dataset["timestamp"])``,
    which worked when a column access returned a list of tensors. `datasets` 5.x
    returns a lazy ``Column`` instead, so the stack raises. The training image
    pins an older `datasets`; this login node has 5.0.1, and downgrading it
    system-wide to read a few clips is the larger change.
    """
    import datasets

    original = datasets.Dataset.__getitem__

    def __getitem__(self, key):  # noqa: N807
        out = original(self, key)
        if isinstance(key, str) and type(out).__name__ == "Column":
            values = list(out)
            if values and not isinstance(values[0], torch.Tensor):
                try:
                    return [torch.as_tensor(v) for v in values]
                except (TypeError, ValueError, RuntimeError):
                    return values
            return values
        return out

    datasets.Dataset.__getitem__ = __getitem__


def patch_torchvision_video_reader() -> None:
    """Restore the `torchvision.io.VideoReader` the vendored LeRobot decoder uses.

    torchvision dropped `VideoReader` after 0.22; this node has 0.27. The decoder
    only needs `seek`, iteration yielding ``{"data": uint8 CHW, "pts": seconds}``
    and a `.container` to close, all of which PyAV provides directly.
    """
    import torchvision

    if hasattr(torchvision.io, "VideoReader"):
        return

    import av

    class VideoReader:
        def __init__(self, path: str, stream: str = "video"):
            del stream
            self.container = av.open(str(path))
            self._stream = self.container.streams.video[0]
            self._stream.thread_type = "AUTO"

        def seek(self, timestamp: float, keyframes_only: bool = True):
            offset = int(max(float(timestamp), 0.0) / self._stream.time_base)
            self.container.seek(offset, stream=self._stream, any_frame=not keyframes_only)
            return self

        def __iter__(self):
            for frame in self.container.decode(self._stream):
                array = torch.from_numpy(frame.to_ndarray(format="rgb24"))
                yield {
                    "data": array.permute(2, 0, 1).contiguous(),
                    "pts": float(frame.pts * self._stream.time_base),
                }

    torchvision.io.VideoReader = VideoReader
    if not hasattr(torchvision, "set_video_backend"):
        torchvision.set_video_backend = lambda backend: None


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    ckpt = args.ckpt or (run_dir / "checkpoints" / "weights" / f"step_{args.step:06d}.pt")
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    os.environ.setdefault(
        "DIFFSYNTH_MODEL_BASE_PATH",
        "./checkpoints",
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import hydra
    from omegaconf import OmegaConf

    from fastwam.utils.logging_config import get_logger

    logger = get_logger("dump_dream_visuals")

    cfg = OmegaConf.load(run_dir / "config.yaml")
    stats = run_dir / "dataset_stats.json"
    if stats.exists():
        # Recomputing min/max over four suites takes minutes and would be a
        # *different* normalisation than the checkpoint was trained with.
        cfg.data.train.pretrained_norm_stats = str(stats)
    # `val_set_proportion: 0.0` means the val split is empty, so the clips have
    # to come from the same split the run trained on; they are still individual
    # held-out *timesteps* the figures were not selected on.
    cfg.data.train.dream_target.enabled = False

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    logger.info("building dataset ...")
    patch_datasets_column_access()
    patch_torchvision_video_reader()
    # RobotVideoDataset re-saves the stats it used into the work dir even when
    # they were loaded from a file, so the work dir has to exist and must not be
    # the run dir (which is the bucket copy the checkpoint was trained with).
    from fastwam.utils import misc

    args.out.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(args.out))
    dataset = hydra.utils.instantiate(cfg.data.train)
    logger.info("dataset: %d clips", len(dataset))

    logger.info("building model ...")
    model = hydra.utils.instantiate(
        cfg.model, model_dtype=torch.bfloat16, device=str(device)
    )
    logger.info("loading %s (%.1f GB) ...", ckpt, ckpt.stat().st_size / 1e9)
    model.load_checkpoint(str(ckpt))
    model.to(device)
    model.eval()

    mot = model.mot
    router = mot.router
    group_ids = router.group_ids.cpu().clone()
    group_names = list(router.group_names)
    logger.info("router groups: %s", group_names)

    if args.indices:
        indices = list(args.indices)
    else:
        n = len(dataset)
        indices = [int(round(i * (n - 1) / max(args.num_samples, 1))) for i in range(args.num_samples)]

    args.out.mkdir(parents=True, exist_ok=True)
    offsets = list(model.dream_expert.future_offsets)
    freq_ratio = int(model.online_action_video_freq_ratio)

    for order, index in enumerate(indices):
        sample = dataset[index]
        batch = {}
        for key, value in sample.items():
            batch[key] = value.unsqueeze(0) if isinstance(value, torch.Tensor) else value

        video = batch["video"].to(device=device, dtype=model.torch_dtype)
        prompt = sample.get("prompt", "")
        logger.info("[%d/%d] index=%d  %s", order + 1, len(indices), index, prompt)

        # --- ground-truth dream targets, from the future frames of this clip ---
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gt = model.online_targets(
                video,
                future_offsets=offsets,
                action_video_freq_ratio=freq_ratio,
                image_is_pad=batch.get("image_is_pad", None),
            )

        # --- deployment forward: video prefill -> dream -> N action steps ------
        mot.capture_action_attention = True
        mot.captured_action_attention = []
        with torch.no_grad():
            out = model.infer_action(
                prompt=None,
                input_image=video[0][:, 0].unsqueeze(0),
                action_horizon=int(batch["action"].shape[1]),
                proprio=batch["proprio"][0, 0] if "proprio" in batch else None,
                context=batch["context"][0],
                context_mask=batch["context_mask"][0],
                num_inference_steps=args.action_steps,
                seed=42,
                return_dream_predictions=True,
            )
        records = mot.captured_action_attention
        mot.capture_action_attention = False
        mot.captured_action_attention = []

        num_layers = int(mot.num_layers)
        if len(records) != num_layers * args.action_steps:
            raise RuntimeError(
                f"expected {num_layers * args.action_steps} captures, got {len(records)}"
            )

        def stack(key: str) -> torch.Tensor:
            """[steps, layers, ...] from the flat capture list."""
            flat = torch.stack([r[key][0] for r in records], dim=0)
            return flat.reshape(args.action_steps, num_layers, *flat.shape[1:])

        payload: dict[str, Any] = {
            "index": index,
            "prompt": prompt,
            "offsets": offsets,
            "group_ids": group_ids,
            "group_names": group_names,
            "num_layers": num_layers,
            "action_steps": args.action_steps,
            # RGB: t0 plus the two futures the dream is asked about.
            "rgb": {
                "t0": video[0][:, 0].float().cpu(),
                **{
                    f"t{i}": video[0][:, offset // freq_ratio].float().cpu()
                    for i, offset in enumerate(offsets)
                },
            },
            "depth_pred": unpatchify_depth(out["dream_predictions"]["depth"].float()).cpu(),
            "depth_gt": unpatchify_depth(gt["depth"][0].float()).cpu(),
            "dino_pred": out["dream_predictions"]["dino"].float().cpu(),
            "dino_gt": gt["dino"][0].float().cpu(),
            "gate": stack("gate"),
            "keep": stack("keep"),
            "dream_probs": stack("dream_probs"),
            # `mot` already averaged over the action queries, so these are
            # [steps, layers]; averaging again would collapse the layer axis.
            "mass_video": stack("mass_video"),
            "mass_dream": stack("mass_dream"),
            "mass_action": stack("mass_action"),
            "action_pred": out["action"].cpu(),
            "action_gt": batch["action"][0].float().cpu(),
            # False means the clip ran out before that offset, so the "ground
            # truth" there is a repeated last frame and must not be scored.
            "future_valid": gt["future_valid_mask"][0].cpu(),
        }
        target = args.out / f"sample_{order:02d}_idx{index}.pt"
        torch.save(payload, target)
        logger.info(
            "  -> %s  keep=%.4f  dream_mass=%.4f  action_l1=%.4f",
            target.name,
            float(payload["keep"].float().mean()),
            float(payload["mass_dream"].mean()),
            float((payload["action_pred"] - payload["action_gt"]).abs().mean()),
        )

    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
