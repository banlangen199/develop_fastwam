"""Construction and training entry points for :class:`RoutedWAM`.

Mirrors :mod:`fastwam.threshold_runtime`: the dense factory in
:mod:`fastwam.runtime` stays the single place where a DreamFastWAM is built and
its config validated, and this module only installs the routed adapter on top.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .models.wan22.routed_wam import RoutedWAM
from .routed_trainer import RoutedWan22Trainer
from .runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
    build_datasets,
    create_dream_fastwam,
)
from .utils import misc
from .utils.logging_config import setup_logging


def _to_container(value):
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def create_routed_wam(
    *,
    router: dict[str, Any] | DictConfig | None = None,
    interface_distill: dict[str, Any] | DictConfig | None = None,
    generative_dream: dict[str, Any] | DictConfig | None = None,
    dream_scheduler: dict[str, Any] | DictConfig | None = None,
    online_dream_targets: dict[str, Any] | DictConfig | None = None,
    finetune_action_only: bool = False,
    training_mode: str = "joint",
    active_dream_modalities=None,
    **dream_fastwam_kwargs,
) -> RoutedWAM:
    """Build a dense DreamFastWAM and promote it to a RoutedWAM."""
    dense_model = create_dream_fastwam(**dream_fastwam_kwargs)
    dense_model.set_active_dream_modalities(active_dream_modalities)
    return RoutedWAM.from_dense_model(
        dense_model,
        router=_to_container(router),
        interface_distill=_to_container(interface_distill),
        generative_dream=_to_container(generative_dream),
        dream_scheduler=_to_container(dream_scheduler),
        online_dream_targets=_to_container(online_dream_targets),
        finetune_action_only=bool(finetune_action_only),
        training_mode=training_mode,
    )


def run_routed_training(cfg: DictConfig) -> None:
    """Standard training setup with router / distillation progress injection."""
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w", encoding="utf-8") as handle:
        OmegaConf.save(config_payload, handle)

    model_device = _resolve_train_device()
    from .utils.pytorch_utils import set_global_seed
    set_global_seed(int(cfg.seed), get_worker_init_fn=False)
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

    trainer = RoutedWan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    before = None
    if unwrapped.training_mode == "router_only" and trainer.accelerator.is_main_process:
        from .utils.routing_experiment import frozen_checksum
        before = frozen_checksum(unwrapped)
    trainer.accelerator.wait_for_everyone()
    trainer.train()
    if before is not None:
        from .utils.routing_experiment import atomic_json
        after = frozen_checksum(unwrapped)
        atomic_json(Path(cfg.output_dir) / "frozen_check.json", {"before": before, "after": after, "unchanged": before == after})
        if before != after:
            raise RuntimeError("Frozen backbone parameters changed during Router-only training.")
