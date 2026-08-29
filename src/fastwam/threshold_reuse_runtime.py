"""Factory for the inference-only threshold-mask reuse adapter."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from .models.wan22.action_dream_threshold_reuse import ThresholdReuseDreamFastWAM
from .runtime import create_dream_fastwam


def create_threshold_reuse_dream_fastwam(
    *,
    action_dream_threshold: dict[str, Any] | DictConfig | None = None,
    action_dream_threshold_reuse: dict[str, Any] | DictConfig | None = None,
    finetune_action_only: bool = False,
    **dream_fastwam_kwargs,
) -> ThresholdReuseDreamFastWAM:
    if isinstance(action_dream_threshold, DictConfig):
        action_dream_threshold = OmegaConf.to_container(action_dream_threshold, resolve=True)
    if isinstance(action_dream_threshold_reuse, DictConfig):
        action_dream_threshold_reuse = OmegaConf.to_container(
            action_dream_threshold_reuse, resolve=True
        )
    dense_model = create_dream_fastwam(**dream_fastwam_kwargs)
    return ThresholdReuseDreamFastWAM.from_dense_model(
        dense_model,
        action_dream_threshold=action_dream_threshold,
        action_dream_threshold_reuse=action_dream_threshold_reuse,
        finetune_action_only=bool(finetune_action_only),
    )
