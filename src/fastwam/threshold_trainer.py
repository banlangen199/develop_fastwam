from __future__ import annotations

from .trainer import Wan22Trainer


class ThresholdWan22Trainer(Wan22Trainer):
    """Wan22Trainer that supplies optimizer-step progress to threshold warmup."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "set_training_progress_provider"):
            raise TypeError("ThresholdWan22Trainer requires ThresholdDreamFastWAM.")
        model.set_training_progress_provider(
            lambda: (
                self.global_step,
                self.max_steps,
                bool(self.log_every > 0 and (self.global_step + 1) % self.log_every == 0),
            )
        )
