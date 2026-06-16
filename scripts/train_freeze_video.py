"""Training entrypoint with video expert frozen.

Usage (same as train_zero1.sh but pointing to this script)::

    CUDA_VISIBLE_DEVICES=4,5 accelerate launch \\
        --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \\
        --num_processes 2 \\
        scripts/train_freeze_video.py \\
        output_dir=./runs/libero_uncond_2cam224_1e-4_300m/<run_id> \\
        task=libero_uncond_2cam224_1e-4_300m \\
        resume=checkpoints/libero_uncond_2cam224_300m.pt

Or use the helper shell script::

    bash scripts/train_freeze_video.sh 2 task=libero_uncond_2cam224_1e-4_300m \\
        resume=checkpoints/libero_uncond_2cam224_300m.pt
"""

import logging

import hydra
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.trainer import Wan22Trainer
from fastwam.utils.config_resolvers import register_default_resolvers

logger = logging.getLogger(__name__)

# ── monkey-patch: freeze video expert during _set_dit_only_train_mode ──────
_original_set_dit_only = Wan22Trainer._set_dit_only_train_mode


def _freeze_video_set_dit_only(self):
    """Same as original but additionally freezes ``mixtures.video`` in the MoT."""
    model = self.accelerator.unwrap_model(self.model)
    model.eval()
    model.requires_grad_(False)
    model.dit.train()
    model.dit.requires_grad_(True)

    if hasattr(model.dit, "mixtures"):
        mixtures = model.dit.mixtures
        # ── freeze video ──
        if "video" in mixtures:
            video_expert = mixtures["video"]
            video_expert.eval()
            video_expert.requires_grad_(False)
            video_params = sum(p.numel() for p in video_expert.parameters())
            logger.info(
                "Video expert FROZEN (eval mode, no grad) — %.2f B params.",
                video_params / 1e9,
            )

        # ── unfreeze action ──
        if "action" in mixtures:
            action_expert = mixtures["action"]
            action_expert.train()
            action_expert.requires_grad_(True)
            action_params = sum(p.numel() for p in action_expert.parameters())
            logger.info(
                "Action expert TRAINABLE — %.2f M params.",
                action_params / 1e6,
            )

    # ── proprio encoder (unchanged from original) ──
    proprio_encoder = getattr(model, "proprio_encoder", None)
    if proprio_encoder is not None:
        proprio_encoder.train()
        proprio_encoder.requires_grad_(True)


Wan22Trainer._set_dit_only_train_mode = _freeze_video_set_dit_only


# ── monkey-patch: exclude video expert from optimizer params ──────────────────
_original_configure_trainable = Wan22Trainer._configure_trainable_parameters


def _freeze_video_configure_trainable(self, model):
    """Same as original but excludes ``mixtures.video`` from returned params."""
    if hasattr(model, "configure_trainable_parameters"):
        return model.configure_trainable_parameters(freeze_video_expert=self.freeze_video_expert)

    model.eval()
    model.requires_grad_(False)
    model.dit.train()
    model.dit.requires_grad_(True)

    # Freeze video expert before collecting params for optimizer
    if hasattr(model.dit, "mixtures") and "video" in model.dit.mixtures:
        video_expert = model.dit.mixtures["video"]
        video_expert.eval()
        video_expert.requires_grad_(False)
        video_params = sum(p.numel() for p in video_expert.parameters())
        logger.info(
            "Video expert FROZEN (no optimizer states) — %.2f B params.",
            video_params / 1e9,
        )

    proprio_encoder = getattr(model, "proprio_encoder", None)
    if proprio_encoder is not None:
        proprio_encoder.train()
        proprio_encoder.requires_grad_(True)

    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        "Trainable parameters (video-excluded): %.3fM",
        sum(p.numel() for p in params) / 1e6,
    )
    return params


Wan22Trainer._configure_trainable_parameters = _freeze_video_configure_trainable

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
