from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..action_dit import ActionDiT
from .dream_query_expert import DreamQueryExpert
from ..fastwam.model import FastWAM
from ..helpers.loader import load_wan22_ti2v_5b_components
from ..mot import MoT
from fastwam.utils.logging_config import get_logger


logger = get_logger(__name__)


class DreamFastWAM(FastWAM):
    """FastWAM variant with a learnable DreamQueryExpert between video/action."""

    def __init__(
        self,
        *args,
        dream_expert: DreamQueryExpert,
        loss_lambda_dream: float = 0.1,
        loss_lambda_dyn: float = 1.0,
        loss_lambda_depth: float = 1.0,
        loss_lambda_dino: float = 1.0,
        loss_lambda_sam: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dream_expert = dream_expert
        self.dream_expert.to(device=self.device, dtype=self.torch_dtype)
        self.loss_lambda_dream = float(loss_lambda_dream)
        self.loss_lambda_dyn = float(loss_lambda_dyn)
        self.loss_lambda_depth = float(loss_lambda_depth)
        self.loss_lambda_dino = float(loss_lambda_dino)
        self.loss_lambda_sam = float(loss_lambda_sam)
        self.param_counts = self._log_expert_param_counts(
            self.video_expert,
            self.dream_expert,
            self.action_expert,
        )

    @classmethod
    def _log_expert_param_counts(cls, video_expert, dream_expert, action_expert):
        counts = {
            "video_expert": sum(p.numel() for p in video_expert.parameters()),
            "dream_expert": sum(p.numel() for p in dream_expert.parameters()),
            "action_expert": sum(p.numel() for p in action_expert.parameters()),
        }
        counts["total"] = counts["video_expert"] + counts["dream_expert"] + counts["action_expert"]
        logger.info(
            "DreamFastWAM parameter counts: video_expert=%.3fM, dream_expert=%.3fM, "
            "action_expert=%.3fM, total=%.3fM",
            counts["video_expert"] / 1e6,
            counts["dream_expert"] / 1e6,
            counts["action_expert"] / 1e6,
            counts["total"] / 1e6,
        )
        target_m = getattr(dream_expert, "target_num_params_m", None)
        actual_m = counts["dream_expert"] / 1e6
        if hasattr(dream_expert, "architecture"):
            dream_expert.architecture["dream_expert"]["actual_num_params_m"] = actual_m
        cls._log_dream_architecture(dream_expert, actual_m)
        if target_m is not None:
            target_m = float(target_m)
            logger.info("Dream expert params: %.1fM, target: %.1fM", actual_m, target_m)
            if target_m > 0 and abs(actual_m - target_m) / target_m > 0.15:
                logger.warning(
                    "Dream expert parameter count differs from target by more than 15%%: actual=%.1fM target=%.1fM",
                    actual_m,
                    target_m,
                )
        return counts

    @classmethod
    def _log_dream_architecture(cls, dream_expert, actual_num_params_m: float):
        if not hasattr(dream_expert, "resolved_architecture"):
            return
        arch = dream_expert.resolved_architecture()
        expert = arch["dream_expert"]
        decoder = arch["dream_decoder"]
        logger.info(
            "Dream expert architecture: hidden_dim=%s ffn_dim=%s num_layers=%s "
            "num_heads=%s attn_head_dim=%s num_dream_tokens=%s "
            "target_num_params_m=%s actual_num_params_m=%.3f",
            expert["hidden_dim"],
            expert["ffn_dim"],
            expert["num_layers"],
            expert["num_heads"],
            expert["attn_head_dim"],
            expert["num_dream_tokens"],
            expert["target_num_params_m"],
            actual_num_params_m,
        )
        logger.info(
            "Dream decoder architecture: decoder_dim=%s decoder_ffn_dim=%s "
            "num_layers=%s num_heads=%s attn_head_dim=%s",
            decoder["decoder_dim"],
            decoder["decoder_ffn_dim"],
            decoder["num_layers"],
            decoder["num_heads"],
            decoder["attn_head_dim"],
        )
        for modality, cfg in decoder["modalities"].items():
            logger.info(
                "Dream decoder modality: %s enabled=%s target_layout=%s target_shape=%s",
                modality,
                cfg["enabled"],
                cfg["target_layout"],
                cfg["target_shape"],
            )

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        dream_query_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_dream: float = 0.1,
        loss_lambda_dyn: float = 1.0,
        loss_lambda_depth: float = 1.0,
        loss_lambda_dino: float = 1.0,
        loss_lambda_sam: float = 1.0,
        action_noise: Optional[dict[str, Any]] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for DreamFastWAM.")
        if dream_query_config is None:
            raise ValueError("`dream_query_config` is required for DreamFastWAM.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for DreamFastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        dream_expert = DreamQueryExpert(**dream_query_config).to(device=device, dtype=torch_dtype)

        for name, expert in (("action", action_expert), ("dream", dream_expert)):
            if int(expert.num_heads) != int(video_expert.num_heads):
                raise ValueError(f"{name} expert `num_heads` must match video expert for MoT mixed attention.")
            if int(expert.attn_head_dim) != int(video_expert.attn_head_dim):
                raise ValueError(f"{name} expert `attn_head_dim` must match video expert for MoT mixed attention.")
            if int(len(expert.blocks)) != int(len(video_expert.blocks)):
                raise ValueError(f"{name} expert `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "dream": dream_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            video_expert=video_expert,
            dream_expert=dream_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            loss_lambda_dream=loss_lambda_dream,
            loss_lambda_dyn=loss_lambda_dyn,
            loss_lambda_depth=loss_lambda_depth,
            loss_lambda_dino=loss_lambda_dino,
            loss_lambda_sam=loss_lambda_sam,
            action_noise=action_noise,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
            "dream_query_expert": "RANDOM_INIT",
        }
        return model

    def freeze_video_expert(self):
        self.video_expert.eval()
        self.video_expert.requires_grad_(False)
        logger.info("Frozen DreamFastWAM video expert parameters.")

    def configure_trainable_parameters(self, freeze_video_expert: bool = False) -> list[torch.nn.Parameter]:
        self.eval()
        self.requires_grad_(False)
        self.mot.train()
        self.action_expert.train()
        self.action_expert.requires_grad_(True)
        self.dream_expert.train()
        self.dream_expert.requires_grad_(True)
        if freeze_video_expert:
            self.freeze_video_expert()
        else:
            self.video_expert.train()
            self.video_expert.requires_grad_(True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train()
            self.proprio_encoder.requires_grad_(True)
        params = [p for p in self.parameters() if p.requires_grad]
        logger.info(
            "DreamFastWAM trainable parameters: %.3fM (freeze_video_expert=%s)",
            sum(p.numel() for p in params) / 1e6,
            freeze_video_expert,
        )
        return params

    def load_checkpoint(self, path, optimizer=None, *, strict_shapes: bool = False):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            current = self.mot.state_dict()
            filtered = {}
            skipped_shape = []
            for key, value in payload["mot"].items():
                if key not in current:
                    filtered[key] = value
                    continue
                if tuple(current[key].shape) != tuple(value.shape):
                    skipped_shape.append((key, tuple(value.shape), tuple(current[key].shape)))
                    continue
                filtered[key] = value
            incompatible = self.mot.load_state_dict(filtered, strict=False)
            missing = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
            logger.info("Loaded DreamFastWAM MoT checkpoint strict=False from %s", path)
            logger.info("Checkpoint missing keys (%d): %s", len(missing), missing[:50])
            logger.info("Checkpoint unexpected keys (%d): %s", len(unexpected), unexpected[:50])
            if skipped_shape:
                logger.warning(
                    "Skipped checkpoint keys with incompatible shapes (%d). First keys: %s",
                    len(skipped_shape),
                    skipped_shape[:20],
                )
            if strict_shapes and (skipped_shape or missing or unexpected):
                raise RuntimeError(
                    "Checkpoint is not structurally compatible with the current DreamFastWAM model. "
                    f"skipped_shape={skipped_shape[:20]} missing={missing[:50]} unexpected={unexpected[:50]}"
                )
            bad_missing = [
                k for k in missing
                if not (k.startswith("mixtures.dream.") or ".dream_" in k or k.startswith("dream_"))
            ]
            if bad_missing:
                logger.warning(
                    "Checkpoint is missing non-dream MoT keys; verify FastWAM backbone loading. First keys: %s",
                    bad_missing[:50],
                )
        elif "dit" in payload:
            if strict_shapes:
                raise RuntimeError(
                    "Cannot strictly load a legacy `dit` checkpoint into DreamFastWAM. "
                    "Use a checkpoint saved with the full `mot` state."
                )
            logger.warning("Loading legacy `dit` checkpoint into DreamFastWAM video expert only.")
            current = self.video_expert.state_dict()
            filtered = {}
            skipped_shape = []
            for key, value in payload["dit"].items():
                if key in current and tuple(current[key].shape) != tuple(value.shape):
                    skipped_shape.append((key, tuple(value.shape), tuple(current[key].shape)))
                    continue
                filtered[key] = value
            incompatible = self.video_expert.load_state_dict(filtered, strict=False)
            if incompatible.missing_keys:
                logger.warning("Video expert missing keys from legacy checkpoint: %s", incompatible.missing_keys[:50])
            if incompatible.unexpected_keys:
                logger.warning("Video expert unexpected keys from legacy checkpoint: %s", incompatible.unexpected_keys[:50])
            if skipped_shape:
                logger.warning("Skipped legacy video keys with incompatible shapes: %s", skipped_shape[:20])
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")

        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def build_inputs(self, sample, tiled: bool = False):
        if self.training and "dream_targets" not in sample:
            raise ValueError(
                "DreamFastWAM training requires non-empty `sample['dream_targets']`. "
                "Enable data.train.dream_target or use model=fastwam for ordinary FastWAM training."
            )
        inputs = super().build_inputs(sample, tiled=tiled)
        if "dream_targets" in sample:
            targets = sample["dream_targets"]
            if not targets:
                raise ValueError("`sample['dream_targets']` is empty; at least one modality is required.")
            mask_keys = {f"{name}_valid_mask" for name in ("dyn", "depth", "dino", "sam")}
            metadata_keys = {"future_valid_mask", "future_offsets", *mask_keys}
            unknown = set(targets.keys()) - {"dyn", "depth", "dino", "sam", *metadata_keys}
            if unknown:
                raise ValueError(f"`sample['dream_targets']` contains unsupported keys: {sorted(unknown)}")
            inputs["dream_targets"] = {
                key: value.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                for key, value in targets.items()
                if key not in metadata_keys
            }
            if "future_valid_mask" in targets:
                inputs["future_valid_mask"] = targets["future_valid_mask"].to(
                    device=self.device, dtype=torch.bool, non_blocking=True
                )
            modality_valid_masks = {}
            for name in ("dyn", "depth", "dino", "sam"):
                key = f"{name}_valid_mask"
                if key in targets:
                    modality_valid_masks[name] = targets[key].to(
                        device=self.device, dtype=torch.bool, non_blocking=True
                    )
            if modality_valid_masks:
                inputs["modality_valid_masks"] = modality_valid_masks
            if "future_offsets" in targets:
                inputs["future_offsets"] = targets["future_offsets"].to(
                    device=self.device, dtype=torch.long, non_blocking=True
                )
        return inputs

    def _assert_training_dream_modalities_match(self, dream_targets: dict[str, torch.Tensor]) -> None:
        target_modalities = set(dream_targets.keys())
        model_modalities = set(getattr(self.dream_expert, "modalities", ("dyn", "depth", "dino", "sam")))
        if target_modalities != model_modalities:
            raise ValueError(
                "Dream target modalities must match enabled dream model modalities during training. "
                f"targets={sorted(target_modalities)} model={sorted(model_modalities)}"
            )

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        dream_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + dream_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        dream_start = video_seq_len
        action_start = video_seq_len + dream_seq_len

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

        current_frame_tokens = min(int(video_tokens_per_frame), int(video_seq_len))
        if current_frame_tokens <= 0:
            raise ValueError(
                f"Cannot build DreamFastWAM attention mask with current_frame_tokens={current_frame_tokens}."
            )

        # Dream/action use only current-frame RGB video tokens during training and inference.
        # This prevents dream supervision from peeking at noisy future video latents and keeps
        # the train-time conditioning contract aligned with single-frame inference.
        mask[dream_start:action_start, :current_frame_tokens] = True
        for slc in self.dream_expert.modality_slices().values():
            rows = slice(dream_start + slc.start, dream_start + slc.stop)
            cols = slice(dream_start + slc.start, dream_start + slc.stop)
            mask[rows, cols] = True

        # action -> current-frame video + all dream + action
        mask[action_start:, :current_frame_tokens] = True
        mask[action_start:, dream_start:action_start] = True
        mask[action_start:, action_start:] = True
        return mask

    def _compute_dream_loss(
        self,
        pred: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        future_valid_mask: torch.Tensor | None = None,
        modality_valid_masks: dict[str, torch.Tensor] | None = None,
        future_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not targets:
            raise ValueError("Dream loss received no target modalities.")
        unknown = set(targets.keys()) - {"dyn", "depth", "dino", "sam"}
        if unknown:
            raise ValueError(f"Dream loss received unsupported target modalities: {sorted(unknown)}")
        for key, target in targets.items():
            if key not in pred:
                raise ValueError(f"Dream decoder did not produce modality={key!r}; available={sorted(pred.keys())}")
            if pred[key].ndim == target.ndim + 1 and pred[key].shape[1] == 1:
                targets[key] = target.unsqueeze(1)
                target = targets[key]
            if tuple(pred[key].shape) != tuple(target.shape):
                raise ValueError(
                    f"Dream prediction/target shape mismatch for {key}: "
                    f"pred={tuple(pred[key].shape)} target={tuple(target.shape)}"
                )

        first_pred = next(iter(pred.values()))
        batch_size = int(first_pred.shape[0])
        num_offsets = int(first_pred.shape[1]) if first_pred.ndim >= 2 else 1
        if future_valid_mask is None:
            future_valid_mask = torch.ones((batch_size, num_offsets), device=first_pred.device, dtype=torch.bool)
        else:
            future_valid_mask = future_valid_mask.to(device=first_pred.device, dtype=torch.bool)
            if future_valid_mask.ndim == 1:
                future_valid_mask = future_valid_mask.unsqueeze(0).expand(batch_size, -1)
            if tuple(future_valid_mask.shape) != (batch_size, num_offsets):
                raise ValueError(
                    f"future_valid_mask shape mismatch: expected {(batch_size, num_offsets)}, "
                    f"got {tuple(future_valid_mask.shape)}"
                )
        valid = future_valid_mask.to(dtype=first_pred.float().dtype)

        def normalize_valid_mask(mask: torch.Tensor | None, name: str) -> torch.Tensor:
            if mask is None:
                return valid
            mask = mask.to(device=first_pred.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(batch_size, -1)
            if tuple(mask.shape) != (batch_size, num_offsets):
                raise ValueError(
                    f"{name}_valid_mask shape mismatch: expected {(batch_size, num_offsets)}, "
                    f"got {tuple(mask.shape)}"
                )
            return mask.to(dtype=first_pred.float().dtype)

        modality_valid_masks = modality_valid_masks or {}
        valid_by_modality = {
            name: normalize_valid_mask(modality_valid_masks.get(name), name)
            for name in ("dyn", "depth", "dino", "sam")
        }

        if future_offsets is None:
            offsets = list(range(num_offsets))
        else:
            if future_offsets.ndim == 2:
                future_offsets = future_offsets[0]
            offsets = [int(x) for x in future_offsets.detach().cpu().tolist()]

        def masked_mean(loss_each: torch.Tensor, modality: str) -> torch.Tensor:
            modality_valid = valid_by_modality[modality]
            return (loss_each * modality_valid).sum() / modality_valid.sum().clamp(min=1.0)

        def reduce_except_batch_horizon(loss: torch.Tensor) -> torch.Tensor:
            if loss.ndim <= 2:
                return loss
            return loss.reshape(loss.shape[0], loss.shape[1], -1).mean(dim=-1)

        zero = first_pred.sum() * 0.0
        loss_dyn = zero
        loss_depth = zero
        loss_dino = zero
        loss_sam = zero
        per_horizon_terms = []
        loss_terms = []
        if "dyn" in targets:
            dyn_min = float(targets["dyn"].detach().amin().item())
            dyn_max = float(targets["dyn"].detach().amax().item())
            if dyn_min < -1e-4 or dyn_max > 1.0 + 1e-4:
                logger.warning(
                    "Dream dyn target should be in [0, 1], got min=%.6f max=%.6f",
                    dyn_min,
                    dyn_max,
                )
            loss_dyn_each = reduce_except_batch_horizon(
                F.binary_cross_entropy_with_logits(pred["dyn"].float(), targets["dyn"].float(), reduction="none")
            )
            loss_dyn = masked_mean(loss_dyn_each, "dyn")
            per_horizon_terms.append((self.loss_lambda_dyn * loss_dyn_each, valid_by_modality["dyn"]))
            loss_terms.append(self.loss_lambda_dyn * loss_dyn)
        if "depth" in targets:
            loss_depth_each = reduce_except_batch_horizon(
                F.smooth_l1_loss(pred["depth"].float(), targets["depth"].float(), reduction="none")
            )
            loss_depth = masked_mean(loss_depth_each, "depth")
            per_horizon_terms.append((self.loss_lambda_depth * loss_depth_each, valid_by_modality["depth"]))
            loss_terms.append(self.loss_lambda_depth * loss_depth)
        if "dino" in targets:
            pred_dino = self._flatten_feature_target(pred["dino"], "dino")
            target_dino = self._flatten_feature_target(targets["dino"], "dino")
            loss_dino_each = 1.0 - F.cosine_similarity(
                F.normalize(pred_dino.float(), dim=-1),
                F.normalize(target_dino.float(), dim=-1),
                dim=-1,
            ).mean(dim=-1)
            loss_dino = masked_mean(loss_dino_each, "dino")
            per_horizon_terms.append((self.loss_lambda_dino * loss_dino_each, valid_by_modality["dino"]))
            loss_terms.append(self.loss_lambda_dino * loss_dino)
        if "sam" in targets:
            pred_sam = self._flatten_feature_target(pred["sam"], "sam")
            target_sam = self._flatten_feature_target(targets["sam"], "sam")
            loss_sam_each = 1.0 - F.cosine_similarity(
                F.normalize(pred_sam.float(), dim=-1),
                F.normalize(target_sam.float(), dim=-1),
                dim=-1,
            ).mean(dim=-1)
            loss_sam = masked_mean(loss_sam_each, "sam")
            per_horizon_terms.append((self.loss_lambda_sam * loss_sam_each, valid_by_modality["sam"]))
            loss_terms.append(self.loss_lambda_sam * loss_sam)
        loss_dream = sum(loss_terms)
        if per_horizon_terms:
            loss_future_num = sum(loss_each * modality_valid for loss_each, modality_valid in per_horizon_terms)
            loss_future_den = sum(modality_valid for _, modality_valid in per_horizon_terms).clamp(min=1.0)
            loss_future_each = loss_future_num / loss_future_den
            report_valid = (sum(modality_valid for _, modality_valid in per_horizon_terms) > 0).to(
                dtype=first_pred.float().dtype
            )
        else:
            loss_future_each = torch.zeros_like(valid)
            report_valid = valid
        parts = {
            "loss_dyn": loss_dyn,
            "loss_depth": loss_depth,
            "loss_dino": loss_dino,
            "loss_sam": loss_sam,
            "future_valid_ratio": valid.mean(),
        }
        for name, modality_valid in valid_by_modality.items():
            if name in targets:
                parts[f"{name}_valid_ratio"] = modality_valid.mean()
        for horizon_idx, offset in enumerate(offsets[:num_offsets]):
            horizon_valid = report_valid[:, horizon_idx]
            horizon_den = horizon_valid.sum().clamp(min=1.0)
            parts[f"future_valid_count_{offset}"] = horizon_valid.sum()
            parts[f"loss_future_{offset}"] = (loss_future_each[:, horizon_idx] * horizon_valid).sum() / horizon_den
        return loss_dream, parts

    @staticmethod
    def _flatten_feature_target(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim == 4:
            return tensor
        if tensor.ndim == 5:
            bsz, num_offsets, h, w, dim = tensor.shape
            return tensor.reshape(bsz, num_offsets, h * w, dim)
        raise ValueError(
            f"{name} dream target must be [B,O,N,C] or [B,O,H,W,C], got {tuple(tensor.shape)}"
        )

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        if "dream_targets" not in inputs:
            raise ValueError(
                "DreamFastWAM training requires non-empty `sample['dream_targets']`."
            )
        dream_targets = inputs["dream_targets"]
        self._assert_training_dream_modalities_match(dream_targets)
        future_valid_mask = inputs.get("future_valid_mask", None)
        modality_valid_masks = inputs.get("modality_valid_masks", None)
        future_offsets = inputs.get("future_offsets", None)

        train_video_branch = self.loss_lambda_video > 0.0
        if train_video_branch:
            noise_video = torch.randn_like(input_latents)
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=input_latents.dtype,
            )
            latents_video = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
            target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
            if inputs["first_frame_latents"] is not None:
                latents_video[:, :, 0:1] = inputs["first_frame_latents"]
        else:
            current_frame_latents = inputs["first_frame_latents"]
            if current_frame_latents is None:
                current_frame_latents = input_latents[:, :, 0:1]
            latents_video = current_frame_latents
            timestep_video = torch.zeros(
                (batch_size,),
                device=self.device,
                dtype=input_latents.dtype,
            )
            target_video = None

        noise_action = self._sample_action_noise(
            action,
            use_correlated_noise=self.use_correlated_noise_train,
        )
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        dream_pre = self.dream_expert.pre_dit(
            batch_size=batch_size,
            device=video_pre["tokens"].device,
            dtype=video_pre["tokens"].dtype,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            dream_seq_len=dream_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "dream": dream_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "dream": dream_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "dream": {"context": dream_pre["context"], "mask": dream_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "dream": dream_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_video = None
        if train_video_branch:
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_dream = self.dream_expert.post_dit(tokens_out["dream"], dream_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        if train_video_branch:
            include_initial_video_step = inputs["first_frame_latents"] is None
            if inputs["first_frame_latents"] is not None:
                pred_video = pred_video[:, :, 1:]
                target_video = target_video[:, :, 1:]
            loss_video_per_sample = self._compute_video_loss_per_sample(
                pred_video=pred_video,
                target_video=target_video,
                image_is_pad=image_is_pad,
                include_initial_video_step=include_initial_video_step,
            )
            video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
            )
            loss_video = (loss_video_per_sample * video_weight).mean()
        else:
            loss_video = pred_action.sum() * 0.0

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_dream, dream_parts = self._compute_dream_loss(
            pred_dream,
            dream_targets,
            future_valid_mask=future_valid_mask,
            modality_valid_masks=modality_valid_masks,
            future_offsets=future_offsets,
        )

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
            + self.loss_lambda_dream * loss_dream
        )
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_dream": self.loss_lambda_dream * float(loss_dream.detach().item()),
            "loss_dyn": self.loss_lambda_dream * self.loss_lambda_dyn * float(dream_parts["loss_dyn"].detach().item()),
            "loss_depth": self.loss_lambda_dream * self.loss_lambda_depth * float(dream_parts["loss_depth"].detach().item()),
            "loss_dino": self.loss_lambda_dream * self.loss_lambda_dino * float(dream_parts["loss_dino"].detach().item()),
            "loss_sam": self.loss_lambda_dream * self.loss_lambda_sam * float(dream_parts["loss_sam"].detach().item()),
            "future_valid_ratio": float(dream_parts["future_valid_ratio"].detach().item()),
        }
        for key, value in dream_parts.items():
            if key.startswith("future_valid_count_"):
                loss_dict[key] = float(value.detach().item())
            elif key.startswith("loss_future_"):
                loss_dict[key] = self.loss_lambda_dream * float(value.detach().item())
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        return_video: bool = False,
        return_dream: bool = False,
        return_action_attention: bool = False,
        attention_layers: Optional[list[int]] = None,
    ):
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        dream_pre = self.dream_expert.pre_dit(
            batch_size=latents_video.shape[0],
            device=video_pre["tokens"].device,
            dtype=video_pre["tokens"].dtype,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        mot_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "dream": dream_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=self._build_mot_attention_mask(
                video_seq_len=video_pre["tokens"].shape[1],
                dream_seq_len=dream_pre["tokens"].shape[1],
                action_seq_len=action_pre["tokens"].shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            ),
            freqs_all={"video": video_pre["freqs"], "dream": dream_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "dream": {"context": dream_pre["context"], "mask": dream_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "dream": dream_pre["t_mod"], "action": action_pre["t_mod"]},
            return_action_attention=return_action_attention,
            attention_layers=attention_layers,
        )
        if return_action_attention:
            tokens_out = mot_out["tokens"]
            attention_records = mot_out["action_attention"]
        else:
            tokens_out = mot_out
            attention_records = None
        pred_video = None
        if return_video:
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_dream = None
        if return_dream:
            pred_dream = self.dream_expert.post_dit(tokens_out["dream"], dream_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        if return_action_attention and return_dream:
            return pred_video, pred_action, pred_dream, attention_records
        if return_action_attention:
            return pred_video, pred_action, attention_records
        if return_dream:
            return pred_video, pred_action, pred_dream
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        return_dream: bool = False,
    ):
        # Full-MoT reference path, retained for joint prediction and cache
        # equivalence checks. `infer_action` uses the optimized cache path.
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        joint_out = self._predict_joint_noise(
            latents_video=first_frame_latents,
            latents_action=latents_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            gt_action=None,
            return_dream=return_dream,
        )
        pred_action = joint_out[1]
        if return_dream:
            return pred_action, joint_out[2]
        return pred_action

    @torch.no_grad()
    def _prefill_video_dream_cache(
        self,
        first_frame_latents: torch.Tensor,
        action_seq_len: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        return_dream: bool = False,
    ) -> dict[str, Any]:
        """Run the action-independent Video and Dream branches once."""
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        dream_pre = self.dream_expert.pre_dit(
            batch_size=first_frame_latents.shape[0],
            device=video_pre["tokens"].device,
            dtype=video_pre["tokens"].dtype,
            context=context,
            context_mask=context_mask,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        dream_seq_len = int(dream_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            dream_seq_len=dream_seq_len,
            action_seq_len=int(action_seq_len),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        context_seq_len = video_seq_len + dream_seq_len
        prefill = self.mot.prefill_video_dream_cache(
            video_tokens=video_pre["tokens"],
            dream_tokens=dream_pre["tokens"],
            video_freqs=video_pre["freqs"],
            dream_freqs=dream_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            dream_t_mod=dream_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            dream_context_payload={
                "context": dream_pre["context"],
                "mask": dream_pre["context_mask"],
            },
            context_attention_mask=attention_mask[:context_seq_len, :context_seq_len],
        )
        dream_predictions = None
        if return_dream:
            dream_predictions = self.dream_expert.post_dit(
                prefill["tokens"]["dream"],
                dream_pre,
            )
        return {
            "kv_cache": prefill["kv_cache"],
            "attention_mask": attention_mask,
            "video_seq_len": video_seq_len,
            "dream_seq_len": dream_seq_len,
            "dream_predictions": dream_predictions,
        }

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_dream_cache: dict[str, Any],
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_context_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            context_kv_cache=video_dream_cache["kv_cache"],
            attention_mask=video_dream_cache["attention_mask"],
            video_seq_len=video_dream_cache["video_seq_len"],
            dream_seq_len=video_dream_cache["dream_seq_len"],
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        return_dream_predictions: bool = False,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action_shape = (1, action_horizon, self.action_expert.action_dim)
        latents_action_base = torch.empty(latents_action_shape, device=rand_device, dtype=torch.float32)
        if self.use_correlated_noise_infer:
            latents_action = self._sample_action_noise(
                latents_action_base,
                use_correlated_noise=True,
                generator=generator,
            ).to(device=self.device, dtype=self.torch_dtype)
        else:
            latents_action = torch.randn(
                latents_action_shape,
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        video_dream_cache = self._prefill_video_dream_cache(
            first_frame_latents=first_frame_latents,
            action_seq_len=latents_action.shape[1],
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
            return_dream=return_dream_predictions,
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        dream_predictions = video_dream_cache["dream_predictions"]
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_dream_cache=video_dream_cache,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        output = {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}
        if return_dream_predictions:
            if dream_predictions is None:
                raise RuntimeError(
                    "Dream predictions were requested but the action inference schedule was empty."
                )
            output["dream_predictions"] = {
                name: value[0].detach().to(device="cpu", dtype=torch.float32)
                for name, value in dream_predictions.items()
            }
            output["future_offsets"] = list(self.dream_expert.future_offsets)
            output["camera_token_split"] = (
                None
                if self.dream_expert.camera_token_split is None
                else list(self.dream_expert.camera_token_split)
            )
        return output
