"""Trainer that supplies optimizer-step progress to the router and distiller.

Deliberately as small as :class:`~fastwam.threshold_trainer.ThresholdWan22Trainer`:
the router warmup, the distillation ramp and the EMA teacher update are all
driven from the progress callback consumed inside ``RoutedWAM.training_loss``,
so the base training loop needs no modification.

It also mirrors every logged scalar into TensorBoard event files.  This repository
only ever had wandb, which is run `offline` on the cluster -- so the router curves
were only readable by scraping the text log, and the text log wraps its metrics
across physical lines (see ``experiments/analysis/parse_train_log.py``).  Writing
events to the run directory, which lives on the shared bucket, makes the curves
openable with a plain ``tensorboard --logdir`` *while the job is still running*,
and keeps working after the job ages out of the cluster's own TensorBoard window.
"""

from __future__ import annotations

import os
from pathlib import Path

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger


logger = get_logger(__name__)


class RoutedWan22Trainer(Wan22Trainer):
    """Wan22Trainer wired to a RoutedWAM's progress-dependent machinery."""

    def _load_weight_checkpoint_before_optimizer(self):
        super()._load_weight_checkpoint_before_optimizer()
        # Only a weight-file warm start resets the router. Full-state resumes
        # and ordinary inference checkpoint loading preserve learned gates.
        if (self.cfg.get("reset_router_on_weight_load", False)
                and self.resume and Path(str(self.resume)).is_file()):
            router = self.model.mot.feature_router
            if router is None or router.full:
                raise ValueError("Router reset requires static or dynamic routing.")
            router.reset_zero()
            logger.info("Reset semantic QK Router after weight load: all initial gates=0 (tanh).")

    def _configure_trainable_parameters(self, model):
        parameters = super()._configure_trainable_parameters(model)
        router_lr = self.cfg.get("router_learning_rate")
        if router_lr is not None and (not math.isfinite(float(router_lr)) or float(router_lr) <= 0):
            raise ValueError("router_learning_rate must be finite and positive, or null.")
        routers = [getattr(model.mot, name, None) for name in ("feature_router", "router")]
        router_ids = {id(p) for router in routers if router is not None for p in router.parameters()}
        # ZeRO flattens each optimizer group. Separate dtypes so FP32 router
        # parameters cannot be packed into the BF16 expert parameter buffer.
        groups = {}
        for parameter in parameters:
            role = "router" if id(parameter) in router_ids else "experts"
            groups.setdefault((role, parameter.dtype), []).append(parameter)
        result = []
        for (role, dtype), values in groups.items():
            lr = float(router_lr) if role == "router" and router_lr is not None else self.learning_rate
            result.append({"params": values, "lr": lr, "name": role})
            logger.info("Optimizer group %s: dtype=%s params=%d peak_lr=%.3e",
                        role, dtype, sum(p.numel() for p in values), lr)
        return result

    def _build_scheduler(self, scheduler_type, total_train_steps, warmup_steps=0):
        if self.cfg.get("router_learning_rate") is None:
            return super()._build_scheduler(scheduler_type, total_train_steps, warmup_steps)
        kind = str(scheduler_type).strip().lower()
        if kind not in {"cosine", "constant"}:
            raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}.")
        total = max(int(total_train_steps), 1)
        warmup = min(max(int(warmup_steps), 0), total - 1)

        def factor(step):
            if warmup and step < warmup:
                return 1 / warmup + (1 - 1 / warmup) * step / warmup
            if kind == "constant":
                return 1.0
            progress = min(max((step - warmup) / (total - warmup), 0.0), 1.0)
            # Each group's minimum is 1% of its own peak, preserving LR ratios.
            return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=factor)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = self.accelerator.unwrap_model(self.model)
        for name in ("feature_router", "router"):
            router = getattr(model.mot, name, None)
            if router is not None and any(p.dtype != torch.float32 for p in router.parameters()):
                raise RuntimeError("Router must remain FP32 after accelerator.prepare().")
        if not hasattr(model, "set_training_progress_provider"):
            raise TypeError("RoutedWan22Trainer requires a RoutedWAM.")
        model.set_training_progress_provider(
            lambda: (
                self.global_step,
                self.max_steps,
                bool(self.log_every > 0 and (self.global_step + 1) % self.log_every == 0),
            )
        )
        self._tb_writer = None
        self._setup_tensorboard()

    # ------------------------------------------------------------ tensorboard
    def _setup_tensorboard(self) -> None:
        """Open a SummaryWriter on the main process, if torch ships one.

        TensorBoard is treated as strictly optional: a missing `tensorboard`
        package must not take a 26-hour training job down with it, so an import
        failure is logged and training continues with the text log as before.
        """
        if not self.accelerator.is_main_process:
            return
        if os.environ.get("FASTWAM_DISABLE_TENSORBOARD", "").lower() in {"1", "true", "yes"}:
            logger.info("TensorBoard logging disabled by FASTWAM_DISABLE_TENSORBOARD.")
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "TensorBoard unavailable (%s); router curves will only be in the text log. "
                "Install `tensorboard` to get event files.",
                exc,
            )
            return
        log_dir = Path(os.environ.get("FASTWAM_TB_DIR") or (Path(self.cfg.output_dir) / "tb"))
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(log_dir))
        except Exception as exc:  # pragma: no cover - bucket/permission dependent
            logger.warning("Could not open TensorBoard writer at %s: %s", log_dir, exc)
            return
        logger.info("TensorBoard events -> %s  (tensorboard --logdir %s)", log_dir, log_dir)

    def _wandb_log(self, payload: dict):
        """Mirror the metric payload into TensorBoard, then log to wandb as usual.

        `_wandb_log` is the single funnel every logged scalar already passes
        through, so hooking it keeps the two sinks in lockstep -- including the
        router statistics, which is the whole point.
        """
        if self._tb_writer is not None:
            for key, value in payload.items():
                # Router group statistics arrive as `router_keep_dino@t1/wrist`;
                # TensorBoard reads `/` as hierarchy, which is what we want, but
                # `@` is left alone so the tag still matches the log text.
                try:
                    self._tb_writer.add_scalar(key, float(value), self.global_step)
                except (TypeError, ValueError):
                    # Non-scalar entries (images, strings) are wandb's business.
                    continue
            self._tb_writer.flush()
        super()._wandb_log(payload)

    def _finish_wandb(self):
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
        super()._finish_wandb()
