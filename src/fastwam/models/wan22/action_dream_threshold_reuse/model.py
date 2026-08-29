from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from fastwam.utils.logging_config import get_logger

from ..action_dream_threshold.model import (
    ActionDreamThresholdMoT,
    ThresholdDreamFastWAM,
)
from ..dream_fastwam.model import DreamFastWAM
from ..wan_video_dit import flash_attention


logger = get_logger(__name__)


@dataclass(frozen=True)
class ActionDreamThresholdReuseConfig:
    """Schedule for inference-only reuse of layer-wise threshold masks."""

    refresh_steps: tuple[int, ...] = (0, 5)

    @classmethod
    def from_dict(
        cls, value: Optional[dict[str, Any]]
    ) -> "ActionDreamThresholdReuseConfig":
        value = {} if value is None else dict(value)
        raw_steps = value.get("refresh_steps", cls.refresh_steps)
        steps = tuple(sorted({int(step) for step in raw_steps}))
        if not steps or steps[0] != 0:
            raise ValueError("action_dream_threshold_reuse.refresh_steps must start with 0.")
        if any(step < 0 for step in steps):
            raise ValueError("action_dream_threshold_reuse.refresh_steps must be non-negative.")
        return cls(refresh_steps=steps)


class ActionDreamThresholdReuseMoT(ActionDreamThresholdMoT):
    """Threshold MoT that refreshes masks sparsely and physically compacts K/V.

    At configured refresh steps this executes the normal dense score pass and
    the thresholded Action attention.  Between refreshes, each layer gathers
    only the previously kept Dream K/V tokens and executes *one* mixed Action
    attention over Video + compact Dream + Action.  The reuse state is scoped
    to a single ``infer_action`` call and is never used for training.
    """

    def __init__(
        self,
        mixtures: Dict[str, torch.nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        action_dream_threshold: Optional[dict[str, Any]] = None,
        action_dream_threshold_reuse: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            mixtures=mixtures,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            action_dream_threshold=action_dream_threshold,
        )
        self.reuse_config = ActionDreamThresholdReuseConfig.from_dict(
            action_dream_threshold_reuse
        )
        self._reuse_step_index: int | None = None
        self._reuse_refresh_steps: set[int] = set()
        self._reuse_layer_masks: list[dict[str, torch.Tensor] | None] = []

    def begin_reuse_inference(self, num_inference_steps: int) -> None:
        """Reset the per-replan masks before the Action denoising loop."""
        total_steps = int(num_inference_steps)
        if total_steps <= 0:
            raise ValueError("num_inference_steps must be positive for mask reuse.")
        self._reuse_step_index = 0
        self._reuse_refresh_steps = {
            step for step in self.reuse_config.refresh_steps if step < total_steps
        }
        # Step 0 is required by config validation and is always in range here.
        self._reuse_layer_masks = [None] * self.num_layers

    def end_reuse_inference(self) -> None:
        self._reuse_step_index = None
        self._reuse_refresh_steps = set()
        self._reuse_layer_masks = []

    def advance_reuse_inference_step(self) -> int:
        if self._reuse_step_index is None:
            raise RuntimeError(
                "Mask-reuse Action attention was invoked outside infer_action. "
                "Call begin_reuse_inference() first."
            )
        step = self._reuse_step_index
        self._reuse_step_index += 1
        return step

    @staticmethod
    def _pack_keep_mask(keep: torch.Tensor) -> dict[str, torch.Tensor]:
        """Store variable per-sample keep sets as padded, ordered indices."""
        if keep.ndim != 2:
            raise ValueError(f"keep mask must be [B,Sd], got {tuple(keep.shape)}")
        counts = keep.sum(dim=-1, dtype=torch.long)
        max_keep = int(counts.max().item())
        if max_keep == 0:
            indices = torch.empty(
                (keep.shape[0], 0), dtype=torch.long, device=keep.device
            )
            valid = torch.empty(
                (keep.shape[0], 0), dtype=torch.bool, device=keep.device
            )
        else:
            # Stable order preserves the original Dream-token order after gather.
            order = torch.argsort(keep.to(dtype=torch.int8), dim=-1, descending=True, stable=True)
            indices = order[:, :max_keep]
            valid = torch.arange(max_keep, device=keep.device).unsqueeze(0) < counts.unsqueeze(1)
        return {"indices": indices, "valid": valid, "counts": counts}

    def _compact_action_attention(
        self,
        *,
        q_action: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        attention_mask: torch.Tensor,
        action_slice: slice,
        context_slices: dict[str, slice],
        packed_keep: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run one Action attention over physically gathered Dream K/V."""
        video_slice = context_slices["video"]
        dream_slice = context_slices["dream"]
        indices = packed_keep["indices"]
        valid = packed_keep["valid"]
        bsz = int(q_action.shape[0])
        if indices.shape[0] != bsz:
            raise ValueError("Cached threshold mask batch size differs from Action batch size.")

        k_video, v_video = k_all[:, video_slice], v_all[:, video_slice]
        k_dream, v_dream = k_all[:, dream_slice], v_all[:, dream_slice]
        k_action, v_action = k_all[:, action_slice], v_all[:, action_slice]
        compact_dream_len = int(indices.shape[1])
        if compact_dream_len:
            gather_index = indices.unsqueeze(-1).expand(-1, -1, k_dream.shape[-1])
            k_dream = torch.gather(k_dream, 1, gather_index)
            v_dream = torch.gather(v_dream, 1, gather_index)
        else:
            k_dream = k_dream[:, :0]
            v_dream = v_dream[:, :0]
        k_compact = torch.cat([k_video, k_dream, k_action], dim=1)
        v_compact = torch.cat([v_video, v_dream, v_action], dim=1)

        action_rows = attention_mask[action_slice, :].to(device=q_action.device)
        video_mask = action_rows[:, video_slice].unsqueeze(0).expand(bsz, -1, -1)
        action_mask = action_rows[:, action_slice].unsqueeze(0).expand(bsz, -1, -1)
        dream_mask_full = action_rows[:, dream_slice].unsqueeze(0).expand(bsz, -1, -1)
        if compact_dream_len:
            dream_indices = indices.unsqueeze(1).expand(-1, action_rows.shape[0], -1)
            dream_mask = torch.gather(dream_mask_full, 2, dream_indices)
            dream_mask &= valid.unsqueeze(1)
        else:
            dream_mask = dream_mask_full[:, :, :0]
        compact_mask = torch.cat([video_mask, dream_mask, action_mask], dim=-1).unsqueeze(1)
        return flash_attention(
            q=q_action,
            k=k_compact,
            v=v_compact,
            num_heads=self.num_heads,
            ctx_mask=compact_mask,
        )

    def _append_reuse_record(
        self,
        *,
        layer_idx: int,
        step_idx: int,
        refreshed: bool,
        context_slices: dict[str, slice],
        action_slice: slice,
        packed_keep: dict[str, torch.Tensor],
    ) -> None:
        cfg = self.threshold_config
        if not (cfg.save_detailed_tensors or (cfg.log_statistics and self._collect_training_statistics)):
            return
        counts = packed_keep["counts"].float()
        slices = dict(context_slices)
        slices["action"] = action_slice
        self._cached_threshold_records.append(
            {
                "layer": int(layer_idx),
                "alpha_current": float(self.current_alpha()),
                "k_mean": float(counts.mean().item()),
                "k_min": int(counts.min().item()),
                "k_max": int(counts.max().item()),
                "slices": {name: [slc.start, slc.stop] for name, slc in slices.items()},
                "reuse_inference_step": int(step_idx),
                "mask_refreshed": bool(refreshed),
            }
        )

    def _action_attention_with_context_cache(
        self,
        *,
        q_action: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        attention_mask: torch.Tensor,
        action_slice: slice,
        context_slices: dict[str, slice],
        layer_idx: int,
    ) -> torch.Tensor:
        cfg = self.threshold_config
        if not cfg.enabled or (self.current_alpha() <= 0.0 and cfg.min_keep_dream_tokens == 0):
            return super()._action_attention_with_context_cache(
                q_action=q_action,
                k_all=k_all,
                v_all=v_all,
                attention_mask=attention_mask,
                action_slice=action_slice,
                context_slices=context_slices,
                layer_idx=layer_idx,
            )
        if self._reuse_step_index is None:
            raise RuntimeError("Threshold reuse is enabled but no inference-step state was initialized.")
        step_idx = int(self._reuse_step_index)
        refresh = step_idx in self._reuse_refresh_steps
        if refresh:
            result = self._threshold_action_attention(
                q_action=q_action,
                k_all=k_all,
                v_all=v_all,
                attention_mask=attention_mask,
                action_slice=action_slice,
                dream_slice=context_slices["dream"],
                alpha=self.current_alpha(),
            )
            action_out, _, keep, _, _, _, _, _ = result
            packed_keep = self._pack_keep_mask(keep)
            self._reuse_layer_masks[layer_idx] = packed_keep
            self._append_reuse_record(
                layer_idx=layer_idx,
                step_idx=step_idx,
                refreshed=True,
                context_slices=context_slices,
                action_slice=action_slice,
                packed_keep=packed_keep,
            )
            return action_out

        packed_keep = self._reuse_layer_masks[layer_idx]
        if packed_keep is None:
            raise RuntimeError(
                f"No cached Dream mask for layer {layer_idx} at denoising step {step_idx}."
            )
        action_out = self._compact_action_attention(
            q_action=q_action,
            k_all=k_all,
            v_all=v_all,
            attention_mask=attention_mask,
            action_slice=action_slice,
            context_slices=context_slices,
            packed_keep=packed_keep,
        )
        self._append_reuse_record(
            layer_idx=layer_idx,
            step_idx=step_idx,
            refreshed=False,
            context_slices=context_slices,
            action_slice=action_slice,
            packed_keep=packed_keep,
        )
        return action_out


class ThresholdReuseDreamFastWAM(ThresholdDreamFastWAM):
    """DreamFastWAM adapter with step-0/5 threshold-mask reuse at inference."""

    @classmethod
    def from_wan22_pretrained(
        cls,
        *args,
        action_dream_threshold: Optional[dict[str, Any]] = None,
        action_dream_threshold_reuse: Optional[dict[str, Any]] = None,
        finetune_action_only: bool = False,
        **kwargs,
    ) -> "ThresholdReuseDreamFastWAM":
        model = DreamFastWAM.from_wan22_pretrained(*args, **kwargs)
        return cls.from_dense_model(
            model,
            action_dream_threshold=action_dream_threshold,
            action_dream_threshold_reuse=action_dream_threshold_reuse,
            finetune_action_only=finetune_action_only,
        )

    @classmethod
    def from_dense_model(
        cls,
        model: DreamFastWAM,
        *,
        action_dream_threshold: Optional[dict[str, Any]] = None,
        action_dream_threshold_reuse: Optional[dict[str, Any]] = None,
        finetune_action_only: bool = False,
    ) -> "ThresholdReuseDreamFastWAM":
        if not isinstance(model, DreamFastWAM):
            raise TypeError(f"Expected DreamFastWAM, got {type(model)}.")
        dense_mot = model.mot
        model.__class__ = cls
        model.finetune_action_only = bool(finetune_action_only)
        model.mot = ActionDreamThresholdReuseMoT(
            mixtures={name: dense_mot.mixtures[name] for name in dense_mot.expert_order},
            mot_checkpoint_mixed_attn=dense_mot.mot_checkpoint_mixed_attn,
            action_dream_threshold=action_dream_threshold,
            action_dream_threshold_reuse=action_dream_threshold_reuse,
        )
        model.dit = model.mot
        model._training_progress_provider = None
        logger.info(
            "Installed inference-only threshold-reuse MoT: enabled=%s alpha=%.4f refresh_steps=%s",
            model.mot.threshold_config.enabled,
            model.mot.threshold_config.alpha,
            model.mot.reuse_config.refresh_steps,
        )
        return model

    def _predict_action_noise_with_cache(self, *args, **kwargs) -> torch.Tensor:
        # Keep the same step index visible to every MoT layer in this Action
        # forward.  Incrementing before the parent call would make the first
        # denoising forward look like step 1, before its layer masks exist.
        try:
            return super()._predict_action_noise_with_cache(*args, **kwargs)
        finally:
            self.mot.advance_reuse_inference_step()

    def infer_action(self, *args, **kwargs):
        num_inference_steps = int(kwargs.get("num_inference_steps", 20))
        self.mot.begin_reuse_inference(num_inference_steps)
        try:
            return super().infer_action(*args, **kwargs)
        finally:
            self.mot.end_reuse_inference()
