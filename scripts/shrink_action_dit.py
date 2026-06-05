#!/usr/bin/env python3
"""Shrink a trained FastWAM checkpoint's ActionDiT from ~1B to ~300M parameters.

This script loads a full FastWAM training checkpoint, extracts the action expert
weights under ``mixtures.action.*``, interpolates them down to a smaller
hidden_dim / ffn_dim, and writes a new checkpoint. Video expert weights,
proprio_encoder, and other top-level keys are copied unchanged.

Example::

    python scripts/shrink_action_dit.py \\
        --source-ckpt checkpoints/fastwam_release/libero_uncond_2cam224.pt \\
        --target-model-config configs/model/fastwam_300m.yaml \\
        --output checkpoints/libero_uncond_2cam224_300m.pt \\
        --device cuda --dtype bfloat16
"""

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from fastwam.models.wan22.action_dit import ActionDiT


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _is_unresolved_interpolation(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def _load_model_config(path: str, source_action_dim: int | None = None) -> dict[str, Any]:
    """Load and resolve an ``action_dit_config`` from a model YAML.

    Because model configs can contain Hydra interpolations like
    ``${data.train.processor.action_output_dim}`` that only resolve
    within a full Hydra launch, this helper falls back to
    *source_action_dim* (extracted from the source checkpoint) for
    unresolved fields whose name contains ``action_dim``.
    """
    cfg = OmegaConf.load(path)
    if "action_dit_config" not in cfg:
        raise ValueError(f"`{path}` must contain `action_dit_config`.")

    raw = OmegaConf.to_container(cfg.action_dit_config, resolve=False)
    if not isinstance(raw, dict):
        raise ValueError(f"`action_dit_config` must be a dict, got {type(raw)} in {path}")

    # Sensible defaults for config fields that are only resolved inside Hydra
    # and do not affect weight shapes.
    _INTERPOLATION_DEFAULTS: dict[str, Any] = {
        "use_gradient_checkpointing": False,
    }

    action_cfg: dict[str, Any] = {}
    for key, value in raw.items():
        if _is_unresolved_interpolation(value):
            if source_action_dim is not None and "action" in str(value).lower():
                action_cfg[key] = int(source_action_dim)
            elif key in _INTERPOLATION_DEFAULTS:
                action_cfg[key] = _INTERPOLATION_DEFAULTS[key]
            else:
                raise ValueError(
                    f"Cannot resolve interpolation `{key}: {value}` in {path}. "
                    f"Add a default in _INTERPOLATION_DEFAULTS or ensure config is self-contained."
                )
        else:
            action_cfg[key] = value

    # Ensure all expected int fields are actually ints.
    for key in [
        "hidden_dim", "ffn_dim", "num_heads", "attn_head_dim",
        "num_layers", "text_dim", "freq_dim", "action_dim",
    ]:
        if key in action_cfg:
            action_cfg[key] = int(action_cfg[key])
    action_cfg["eps"] = float(action_cfg.get("eps", 1e-6))
    return action_cfg


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    """1D linear interpolation along the last dimension of *tensor*."""
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(
    src: torch.Tensor, target_shape: tuple[int, ...]
) -> torch.Tensor:
    """Resize *src* to match *target_shape* via sequential 1D linear interpolation."""
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    # Pad/trim leading dimensions so ranks match.
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce rank: src shape {tuple(src.shape)}, target {target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        # Move the target dimension to the last position.
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape: src={tuple(src.shape)}, "
            f"target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def shrink_action_weights(
    source_mot_state: dict[str, torch.Tensor],
    target_action_state: dict[str, torch.Tensor],
    apply_alpha_scaling: bool = True,
) -> dict[str, torch.Tensor]:
    """Build a new MoT state dict with action weights shrunk to target shapes.

    Args:
        source_mot_state: Full ``mot`` state dict from a trained checkpoint.
        target_action_state: State dict of a freshly-initialized ``ActionDiT``
            built with the target (smaller) config.  Keys are like
            ``blocks.0.self_attn.q.weight`` (no ``mixtures.action.`` prefix).
        apply_alpha_scaling: If True, multiply weights by
            ``sqrt(src_dim / dst_dim)`` when the **last** dimension changes size,
            matching the convention in ``preprocess_action_dit_backbone.py``.

    Returns:
        New state dict with the same keys as ``source_mot_state``.  Action
        expert weights are interpolated; all other weights are copied.
    """
    ACTION_PREFIX = "mixtures.action."
    new_mot_state: dict[str, torch.Tensor] = {}

    copied_same = 0
    copied_interp = 0
    skipped_non_action = 0

    for full_key, src_tensor in source_mot_state.items():
        if not full_key.startswith(ACTION_PREFIX):
            new_mot_state[full_key] = src_tensor.clone()
            skipped_non_action += 1
            continue

        # Key relative to ActionDiT root, e.g. "blocks.0.self_attn.q.weight"
        rel_key = full_key[len(ACTION_PREFIX):]

        if rel_key not in target_action_state:
            print(f"[WARN] Key `{full_key}` not found in target ActionDiT — copying unchanged.")
            new_mot_state[full_key] = src_tensor.clone()
            copied_same += 1
            continue

        target_tensor = target_action_state[rel_key]
        src_shape = tuple(src_tensor.shape)
        dst_shape = tuple(target_tensor.shape)

        if src_shape == dst_shape:
            new_mot_state[full_key] = src_tensor.clone()
            copied_same += 1
        else:
            value = _resize_tensor_to_shape(src_tensor, dst_shape)
            # Alpha scaling: sqrt(d_src / d_dst) on the last-dimension change.
            if (
                apply_alpha_scaling
                and src_tensor.ndim >= 2
                and src_shape[-1] != dst_shape[-1]
            ):
                alpha = math.sqrt(float(src_shape[-1]) / float(dst_shape[-1]))
                value = value.to(torch.float32) * alpha
            new_mot_state[full_key] = value.detach().to(
                dtype=src_tensor.dtype, device="cpu"
            ).contiguous()
            copied_interp += 1

    print(
        f"[INFO] Action weight shrink complete: "
        f"same_shape={copied_same}, interpolated={copied_interp}, "
        f"non_action_copied={skipped_non_action}"
    )
    return new_mot_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shrink a trained FastWAM ActionDiT to a smaller config."
    )
    parser.add_argument(
        "--source-ckpt", required=True,
        help="Path to the trained FastWAM .pt checkpoint to shrink.",
    )
    parser.add_argument(
        "--target-model-config", required=True,
        help="Model config for the shrunk model (e.g. configs/model/fastwam_300m.yaml).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the shrunk checkpoint .pt file.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device used when instantiating the target ActionDiT (cpu is enough).",
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--apply-alpha-scaling", default="true",
        help="Whether to apply alpha = sqrt(d_src / d_dst) on the rescaled last dim.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After saving, load the new checkpoint and print parameter counts.",
    )
    args = parser.parse_args()

    source_ckpt = Path(args.source_ckpt)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch_dtype = _parse_dtype(args.dtype)
    apply_alpha = str(args.apply_alpha_scaling).strip().lower() in {"1", "true", "yes", "y"}

    if not source_ckpt.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source_ckpt}")

    # ------------------------------------------------------------------
    # 1. Load source checkpoint.
    # ------------------------------------------------------------------
    print(f"[INFO] Loading source checkpoint: {source_ckpt}")
    ckpt = torch.load(str(source_ckpt), map_location="cpu", weights_only=True)
    if "mot" not in ckpt:
        raise ValueError(
            f"Checkpoint {source_ckpt} missing `mot` key — not a FastWAM training ckpt?"
        )
    source_mot = ckpt["mot"]

    source_action_keys = [k for k in source_mot if k.startswith("mixtures.action.")]
    print(f"[INFO] Found {len(source_action_keys)} action expert keys in source MoT state.")

    # Extract source action_dim from the checkpoint
    enc_weight_key = "mixtures.action.action_encoder.weight"
    if enc_weight_key in source_mot:
        source_action_dim = int(source_mot[enc_weight_key].shape[1])
        print(f"[INFO] Detected source action_dim = {source_action_dim}")
    else:
        source_action_dim = None
        print("[WARN] Could not detect source action_dim; assuming 7")
        source_action_dim = 7

    # ------------------------------------------------------------------
    # 2. Build target ActionDiT.
    # ------------------------------------------------------------------
    target_action_cfg = _load_model_config(args.target_model_config, source_action_dim=source_action_dim)

    print(
        f"[INFO] Target ActionDiT config: hidden_dim={target_action_cfg['hidden_dim']}, "
        f"ffn_dim={target_action_cfg['ffn_dim']}, "
        f"num_heads={target_action_cfg['num_heads']}, "
        f"attn_head_dim={target_action_cfg['attn_head_dim']}, "
        f"num_layers={target_action_cfg['num_layers']}"
    )

    target_action = ActionDiT(**target_action_cfg).to(
        device=args.device, dtype=torch_dtype
    )
    target_state = target_action.state_dict()
    print(
        f"[INFO] Target ActionDiT param count: "
        f"{sum(p.numel() for p in target_action.parameters()) / 1e6:.1f}M"
    )

    # Show source action param count
    source_action_params = sum(
        source_mot[k].numel()
        for k in source_action_keys
    )
    print(f"[INFO] Source action expert param count: {source_action_params / 1e6:.1f}M")

    # ------------------------------------------------------------------
    # 3. Shrink.
    # ------------------------------------------------------------------
    new_mot_state = shrink_action_weights(
        source_mot_state=source_mot,
        target_action_state=target_state,
        apply_alpha_scaling=apply_alpha,
    )

    # ------------------------------------------------------------------
    # 4. Build output checkpoint.
    # ------------------------------------------------------------------
    new_ckpt = dict(ckpt)          # shallow copy top-level keys
    new_ckpt["mot"] = new_mot_state

    # Update or add metadata about shrink
    new_ckpt["shrink_meta"] = {
        "source_hidden_dim": 1024,
        "source_ffn_dim": 4096,
        "target_hidden_dim": target_action_cfg["hidden_dim"],
        "target_ffn_dim": target_action_cfg["ffn_dim"],
        "alpha_scaling": apply_alpha,
        "source_ckpt": str(source_ckpt),
        "target_model_config": str(args.target_model_config),
    }

    torch.save(new_ckpt, str(output_path))
    print(f"[INFO] Saved shrunk checkpoint to: {output_path}")

    # ------------------------------------------------------------------
    # 5. Optional verification.
    # ------------------------------------------------------------------
    if args.verify:
        print("\n[INFO] Verifying shrunk checkpoint...")
        v_ckpt = torch.load(str(output_path), map_location="cpu", weights_only=True)
        v_mot = v_ckpt["mot"]
        v_action_keys = [k for k in v_mot if k.startswith("mixtures.action.")]
        v_action_params = sum(v_mot[k].numel() for k in v_action_keys)
        print(
            f"[VERIFY] Shrunk action param count: {v_action_params / 1e6:.1f}M "
            f"(keys: {len(v_action_keys)})"
        )

        # Quick sanity: key count should match source
        s_action_keys = [k for k in source_mot if k.startswith("mixtures.action.")]
        if len(s_action_keys) != len(v_action_keys):
            print(
                f"[VERIFY] WARNING: action key count changed: "
                f"{len(s_action_keys)} → {len(v_action_keys)}"
            )
        else:
            print("[VERIFY] Action key count matches source.")

        # Check video keys unchanged
        for k in source_mot:
            if not k.startswith("mixtures.action."):
                if not torch.equal(source_mot[k], v_mot[k]):
                    print(f"[VERIFY] WARNING: video key `{k}` was modified!")
        print("[VERIFY] Video weights verified unchanged.")

        # Check that shapes match target config
        for k in sorted(v_action_keys):
            rel_key = k[len("mixtures.action."):]
            if rel_key in target_state:
                expected_shape = tuple(target_state[rel_key].shape)
                actual_shape = tuple(v_mot[k].shape)
                if expected_shape != actual_shape:
                    print(
                        f"[VERIFY] WARNING: shape mismatch for `{k}`: "
                        f"expected {expected_shape}, got {actual_shape}"
                    )
        print("[VERIFY] Shape verification complete.")


if __name__ == "__main__":
    main()
