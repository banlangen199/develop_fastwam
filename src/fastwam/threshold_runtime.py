from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .models.wan22.action_dream_threshold import ThresholdDreamFastWAM
from .runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
    build_datasets,
    create_dream_fastwam,
)
from .threshold_trainer import ThresholdWan22Trainer
from .utils import misc
from .utils.logging_config import setup_logging


def create_threshold_dream_fastwam(
    *,
    action_dream_threshold: dict[str, Any] | DictConfig | None = None,
    finetune_action_only: bool = False,
    **dream_fastwam_kwargs,
) -> ThresholdDreamFastWAM:
    """Create the threshold adapter without changing the dense factory."""
    if isinstance(action_dream_threshold, DictConfig):
        action_dream_threshold = OmegaConf.to_container(action_dream_threshold, resolve=True)
    dense_model = create_dream_fastwam(**dream_fastwam_kwargs)
    return ThresholdDreamFastWAM.from_dense_model(
        dense_model,
        action_dream_threshold=action_dream_threshold,
        finetune_action_only=bool(finetune_action_only),
    )


def run_threshold_training(cfg: DictConfig) -> None:
    """The standard training setup with threshold progress injection."""
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w", encoding="utf-8") as handle:
        OmegaConf.save(config_payload, handle)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)

    action_noise_cfg = cfg.model.get("action_noise", {})
    if bool(action_noise_cfg.get("use_correlated_noise_train", False)) or bool(
        action_noise_cfg.get("use_correlated_noise_infer", False)
    ):
        stats_path = action_noise_cfg.get("dataset_stats_path") or os.path.join(
            misc.get_work_dir(), "dataset_stats.json"
        )
        model.load_action_noise_stats(
            stats_path,
            action_key=str(action_noise_cfg.get("action_key", "default")),
        )

    trainer = ThresholdWan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()
