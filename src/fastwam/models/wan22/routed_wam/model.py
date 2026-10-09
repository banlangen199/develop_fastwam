"""RoutedWAM: direct Dream regression with Video/Action-conditioned group routing.

The default Dream expert predicts future targets in one forward, without Dream
noise or denoising. Action retains its flow-matching objective. The optional
generative and distillation paths remain available for legacy experiments:

``generative``
    The Dream expert denoises its multi-modal future targets instead of
    regressing them (:mod:`generative_dream`).

``router``
    Per-layer sigmoid Dream group priors use Video + Action (VA), Dream + Action
    (DA), or input-independent learned logits (S). The existing shared
    mixed attention receives +log(gate) on Dream logits (:mod:`router`).
    The earlier full/static/dynamic QK scaling experiments remain selectable.

``interface_distill``
    A one-step Dream pass is trained to reproduce the per-layer Dream K/V of an
    N-step EMA teacher, so the multi-step imagination collapses into a single
    forward whose output is cached and reused across every action denoising step
    (:mod:`interface_distill`).

With all three disabled the class is behaviourally identical to its parent, and
``tests/test_routed_wam.py`` asserts that element-wise.
"""

from __future__ import annotations

from typing import Any, Optional
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from ..dream_fastwam.model import DreamFastWAM
from ..fastwam.model import FastWAM
from ..schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .generative_dream import GenerativeDreamExpert
from .interface_distill import InterfaceDistillConfig, InterfaceDistiller
from .mot import RoutedMoT
from .online_targets import OnlineDreamTargets, OnlineTargetConfig
from .router import ImaginationRouter, RouterConfig, build_group_ids


logger = get_logger(__name__)

# State-dict prefixes introduced by this module. They are legitimately absent
# from any DreamFastWAM checkpoint, so `load_checkpoint` must not treat them as
# a structural incompatibility.
NEW_PARAMETER_PREFIXES = (
    "router.",
    "feature_router.",
    "mixtures.dream.target_encoders.",
    "mixtures.dream.decoder_conditioners.",
    "mixtures.dream.dream_time_embedding.",
    "mixtures.dream.dream_time_projection.",
)


def _is_new_parameter(key: str) -> bool:
    return any(key.startswith(prefix) or f".{prefix}" in key for prefix in NEW_PARAMETER_PREFIXES)


class RoutedWAM(DreamFastWAM):
    """DreamFastWAM with routed imagination and optional legacy generation."""

    # ------------------------------------------------------------ construction
    @classmethod
    def from_wan22_pretrained(
        cls,
        *args,
        router: Optional[dict[str, Any]] = None,
        interface_distill: Optional[dict[str, Any]] = None,
        generative_dream: Optional[dict[str, Any]] = None,
        dream_scheduler: Optional[dict[str, Any]] = None,
        online_dream_targets: Optional[dict[str, Any]] = None,
        finetune_action_only: bool = False,
        training_mode: str = "joint",
        **kwargs,
    ) -> "RoutedWAM":
        model = DreamFastWAM.from_wan22_pretrained(*args, **kwargs)
        return cls.from_dense_model(
            model,
            router=router,
            interface_distill=interface_distill,
            generative_dream=generative_dream,
            dream_scheduler=dream_scheduler,
            online_dream_targets=online_dream_targets,
            finetune_action_only=finetune_action_only,
            training_mode=training_mode,
        )

    @classmethod
    def from_dense_model(
        cls,
        model: DreamFastWAM,
        *,
        router: Optional[dict[str, Any]] = None,
        interface_distill: Optional[dict[str, Any]] = None,
        generative_dream: Optional[dict[str, Any]] = None,
        dream_scheduler: Optional[dict[str, Any]] = None,
        online_dream_targets: Optional[dict[str, Any]] = None,
        finetune_action_only: bool = False,
        training_mode: str = "joint",
    ) -> "RoutedWAM":
        if not isinstance(model, DreamFastWAM):
            raise TypeError(f"Expected DreamFastWAM, got {type(model)}.")

        router_config = RouterConfig.from_dict(router)
        distill_config = InterfaceDistillConfig.from_dict(interface_distill)
        generative_config = dict(generative_dream or {})
        online_config = OnlineTargetConfig.from_dict(online_dream_targets)

        model.__class__ = cls
        model.router_config = router_config
        model.distill_config = distill_config
        model.finetune_action_only = bool(finetune_action_only)
        if training_mode not in {"joint", "dense_joint", "router_only"}:
            raise ValueError("training_mode must be joint, dense_joint, or router_only.")
        if training_mode != "joint" and (generative_config.get("enabled", False) or distill_config.enabled):
            raise ValueError("Dense experiment training modes cannot enable generation or distillation.")
        model.training_mode = training_mode
        model.routing_eval_policy = "native"
        model.routing_eval_gates = None
        model._routing_gradient_norms = {}
        model._training_progress_provider = None
        model._routing_global_step = 0
        model._last_ema_step = -1

        # 1. Promote the Dream expert in place so the dense factory is reused.
        GenerativeDreamExpert.promote(model.dream_expert, generative_config)
        model.dream_expert.to(device=model.device, dtype=model.torch_dtype)

        # 2. Build the router over the Dream token layout.
        dense_mot = model.mot
        dream_expert = model.dream_expert
        group_ids, group_names = build_group_ids(
            modalities=list(dream_expert.modalities),
            num_future_offsets=int(dream_expert.num_future_offsets),
            tokens_per_modality={
                name: int(getattr(dream_expert, f"n_{name}")) for name in dream_expert.modalities
            },
            granularity=router_config.group_granularity,
            camera_token_split=dream_expert.camera_token_split,
        )
        imagination_router = (
            ImaginationRouter(
                config=router_config,
                num_layers=dense_mot.num_layers,
                inner_dim=dense_mot.num_heads * dense_mot.attn_head_dim,
                num_dream_tokens=int(dream_expert.num_dream_tokens),
                group_ids=group_ids,
                group_names=group_names,
                future_offsets=list(dream_expert.future_offsets),
            )
            if router_config.enabled
            else None
        )

        # 3. Swap in the routed MoT, keeping the very same expert modules.
        model.mot = RoutedMoT(
            mixtures={name: dense_mot.mixtures[name] for name in dense_mot.expert_order},
            mot_checkpoint_mixed_attn=dense_mot.mot_checkpoint_mixed_attn,
            router=imagination_router,
        )
        model.dit = model.mot
        if imagination_router is not None:
            model.mot.router.to(device=model.device, dtype=model.torch_dtype)
        model.mot.feature_router = None
        if router_config.routing_mode is not None:
            model.mot.feature_router = DynamicFeatureRouter(
                config=router_config,
                input_dim=dense_mot.num_heads * dense_mot.attn_head_dim
                + model.text_dim + (model.proprio_dim or 0),
                dream_expert=dream_expert,
            ).to(device=model.device, dtype=model.torch_dtype)
            for name, parameter in model.mot.feature_router.named_parameters():
                if parameter.requires_grad:
                    def record_gradient(gradient, name=name):
                        model._routing_gradient_norms[name] = gradient.detach().float().norm()
                    parameter.register_hook(record_gradient)

        # 4. Dream diffusion scheduler (only consulted in generative mode).
        scheduler_kwargs = dict(dream_scheduler or {})
        model.train_dream_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(scheduler_kwargs.get("num_train_timesteps", 1000)),
            shift=float(scheduler_kwargs.get("train_shift", 5.0)),
        ) if model.dream_expert.generative_enabled else None
        model.infer_dream_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(scheduler_kwargs.get("num_train_timesteps", 1000)),
            shift=float(scheduler_kwargs.get("infer_shift", 5.0)),
        ) if model.dream_expert.generative_enabled else None
        model.dream_inference_steps = int(scheduler_kwargs.get("inference_steps", 1))
        if model.dream_inference_steps < 1:
            raise ValueError("dream_scheduler.inference_steps must be >= 1.")

        # 5. Interface distillation teacher.
        model.distiller = (
            InterfaceDistiller(
                config=distill_config,
                dream_expert=model.dream_expert,
                num_layers=model.mot.num_layers,
            )
            if distill_config.enabled
            else None
        )
        if model.distiller is not None:
            model.distiller.to(device=model.device, dtype=model.torch_dtype)
            if not model.dream_expert.generative_enabled:
                raise ValueError(
                    "interface_distill.enabled=true requires generative_dream.enabled=true: "
                    "with a regression Dream there is only one step and nothing to distil."
                )

        # 5b. Online dream targets. The configured future offsets land on
        # frames the clip already contains (video_sample_indices), so the
        # targets can be extracted on device instead of being read back from
        # tens of GB of precomputed `extras/`.
        model.online_target_config = online_config
        model.online_targets = OnlineDreamTargets(online_config) if online_config.enabled else None
        # Frames kept per video frame; 4 for LIBERO (33 raw -> 9 video frames).
        model.online_action_video_freq_ratio = 4
        if model.online_targets is not None:
            model.online_targets.to(device=model.device, dtype=model.torch_dtype)

        logger.info(
            "Installed RoutedWAM: router=%s generative_dream=%s interface_distill=%s "
            "dream_inference_steps=%d action_only=%s",
            router_config.routing_mode or router_config.mode,
            model.dream_expert.generative_enabled,
            distill_config.enabled,
            model.dream_inference_steps,
            model.finetune_action_only,
        )
        return model

    # -------------------------------------------------------------- bookkeeping
    @property
    def uses_split_path(self) -> bool:
        """Whether training should run Video-once / Dream-per-step / Action-cached.

        This is the deployment computation, so training through it removes a
        train/test mismatch; it is mandatory for interface distillation, which
        needs the per-layer Dream K/V that only this path materialises.
        """
        return (
            bool(self.distill_config.enabled)
            or bool(self.dream_expert.generative_enabled)
            or self.online_targets is not None
        )

    def set_training_progress_provider(self, provider) -> None:
        self._training_progress_provider = provider

    def _refresh_progress(self) -> None:
        if self._training_progress_provider is None:
            return
        progress = tuple(self._training_progress_provider())
        if len(progress) == 2:
            global_step, total_steps = progress
        elif len(progress) == 3:
            global_step, total_steps, _ = progress
        else:
            raise ValueError("Training progress provider must return 2 or 3 values.")
        total_steps = max(int(total_steps), 0)
        global_step = max(int(global_step), 0)
        self._routing_global_step = global_step

        feature_router = getattr(self.mot, "feature_router", None)
        if feature_router is not None:
            feature_router.global_step = global_step

        if self.mot.router is not None:
            warmup = self.router_config.warmup_ratio
            if warmup <= 0.0 or total_steps <= 0:
                self.mot.router.set_progress(1.0)
            else:
                warmup_steps = max(int(round(total_steps * warmup)), 1)
                self.mot.router.set_progress(min(global_step / warmup_steps, 1.0))
        if self.distiller is not None:
            warmup = self.distill_config.warmup_ratio
            if warmup <= 0.0 or total_steps <= 0:
                self.distiller.set_progress(1.0)
            else:
                warmup_steps = max(int(round(total_steps * warmup)), 1)
                self.distiller.set_progress(min(global_step / warmup_steps, 1.0))
            # Advance the EMA teacher once per optimizer step. `training_loss`
            # is called once per micro-batch, so keying on the global step keeps
            # the decay schedule independent of gradient accumulation -- and it
            # avoids having to override the trainer's inner loop.
            if self.training and global_step != self._last_ema_step:
                self._last_ema_step = global_step
                self.distiller.update_ema(self.dream_expert)

    def configure_trainable_parameters(self, freeze_video_expert: bool = False):
        if self.training_mode == "router_only":
            router = self.mot.feature_router
            if router is None or router.full or self.finetune_action_only:
                raise ValueError("router_only requires a trainable semantic Router and no action-only override.")
            self.eval()
            self.requires_grad_(False)
            router.train().requires_grad_(True)
            return list(router.parameters())
        if self.training_mode == "dense_joint" and not freeze_video_expert:
            raise ValueError("dense_joint requires freeze_video_expert=true.")
        if self.finetune_action_only:
            self.eval()
            self.requires_grad_(False)
            self.mot.train()
            self.action_expert.train()
            self.action_expert.requires_grad_(True)
            if self.mot.router is not None:
                self.mot.router.train()
                self.mot.router.requires_grad_(True)
            if getattr(self.mot, "feature_router", None) is not None:
                self.mot.feature_router.train()
                self.mot.feature_router.requires_grad_(not self.mot.feature_router.full)
            params = [p for p in self.parameters() if p.requires_grad]
            logger.info(
                "RoutedWAM action-only fine-tuning: %.3fM trainable parameters.",
                sum(p.numel() for p in params) / 1e6,
            )
            return params

        params = super().configure_trainable_parameters(freeze_video_expert=freeze_video_expert)
        if getattr(self.mot, "feature_router", None) is not None:
            self.mot.feature_router.train()
            self.mot.feature_router.requires_grad_(not self.mot.feature_router.full)
            params = [p for p in self.parameters() if p.requires_grad]
        if self.mot.router is not None:
            # The parent freezes everything and then re-enables the experts it
            # knows about; the router is new, so it must be re-enabled here or it
            # would silently never receive gradients.
            self.mot.router.train()
            self.mot.router.requires_grad_(True)
            params = [p for p in self.parameters() if p.requires_grad]
        if self.distiller is not None:
            # The EMA teacher is never optimised; it follows the student.
            self.distiller.teacher_dream.requires_grad_(False)
            self.distiller.teacher_dream.eval()
            params = [p for p in params if p.requires_grad]
        if self.distill_config.enabled and not freeze_video_expert:
            raise ValueError(
                "interface_distill requires freeze_video_expert=true. The split "
                "Video-once prefill is only valid while the Video expert's K/V are "
                "independent of the diffusion step, which a trainable Video expert "
                "would break."
            )
        logger.info(
            "RoutedWAM trainable parameters: %.3fM (router=%s, distill=%s).",
            sum(p.numel() for p in params) / 1e6,
            self.router_config.routing_mode or self.router_config.mode,
            self.distill_config.enabled,
        )
        return params

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "training_mode", None) == "router_only":
            # Calls from DDP, Trainer, or a caller must not reactivate backbone dropout.
            for module in (self.video_expert, self.dream_expert, self.action_expert,
                           self.proprio_encoder, self.vae, self.text_encoder):
                if module is not None:
                    module.eval()
            self.mot.feature_router.train(mode)
        return self

    def load_checkpoint(self, path, optimizer=None, *, strict_shapes: bool = False):
        """Load a DreamFastWAM checkpoint, tolerating this module's new keys.

        The parent raises under ``strict_shapes`` on *any* missing key, and
        LIBERO evaluation passes ``strict_shapes=True``.  Router and generative
        Dream parameters are legitimately absent from a dense checkpoint, so the
        strictness check is re-run here with those keys excluded -- everything
        pretrained stays strict.
        """
        if strict_shapes:
            report = self.verify_checkpoint_compatibility(path)
            if report["missing_new_parameters"]:
                logger.info(
                    "Checkpoint lacks %d new RoutedWAM parameters; these keep their "
                    "initialisation. First keys: %s",
                    len(report["missing_new_parameters"]),
                    report["missing_new_parameters"][:20],
                )
        payload = super().load_checkpoint(path, optimizer=optimizer, strict_shapes=False)
        local_decoder_prefix = "mixtures.dream.decoder_conditioners."
        if any(
            key.startswith(local_decoder_prefix) and key not in payload.get("mot", {})
            for key in self.mot.state_dict()
        ):
            logger.warning(
                "Checkpoint lacks noise-conditioned Dream decoder weights. The new "
                "local noise path is initialized and needs training before evaluation."
            )
        return payload

    def verify_checkpoint_compatibility(self, path) -> dict[str, list[str]]:
        """Report which keys a checkpoint is missing, split into old and new.

        Used by evaluation in place of the parent's ``strict_shapes`` flag: new
        parameters may be missing, pretrained ones may not.
        """
        payload = torch.load(path, map_location="cpu")
        if "mot" not in payload:
            raise ValueError(f"Checkpoint has no `mot` state: {path}")
        current = self.mot.state_dict()
        missing = [k for k in current if k not in payload["mot"]]
        unexpected = [k for k in payload["mot"] if k not in current]
        shape_mismatch = [
            k
            for k, v in payload["mot"].items()
            if k in current and tuple(current[k].shape) != tuple(v.shape)
        ]
        pretrained_missing = [k for k in missing if not _is_new_parameter(k)]
        if pretrained_missing or unexpected or shape_mismatch:
            raise RuntimeError(
                "Checkpoint is not compatible with this RoutedWAM. "
                f"missing_pretrained={pretrained_missing[:20]} "
                f"unexpected={unexpected[:20]} shape_mismatch={shape_mismatch[:20]}"
            )
        return {
            "missing_new_parameters": [k for k in missing if _is_new_parameter(k)],
            "unexpected": unexpected,
        }

    # --------------------------------------------------------------- dream I/O
    def _dream_target_reference(self, targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        expected = set(self.dream_expert.modalities)
        got = set(targets.keys())
        if not expected.issubset(got):
            raise ValueError(
                f"Dream targets are missing modalities {sorted(expected - got)}; got {sorted(got)}."
            )
        return {name: targets[name] for name in self.dream_expert.modalities}

    def _sample_dream_noise(self, targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: torch.randn_like(value.float()).to(value.dtype) for name, value in targets.items()}

    def _dream_noise_like_targets(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Pure noise in target space, for inference where no target exists."""
        noise: dict[str, torch.Tensor] = {}
        offsets = int(self.dream_expert.num_future_offsets)
        for name in self.dream_expert.modalities:
            decoder = self.dream_expert.decoders[name]
            if not bool(getattr(decoder, "enabled", True)):
                continue
            shape = (batch_size, offsets, *decoder.target_shape)
            noise[name] = torch.randn(shape, device=device, dtype=torch.float32, generator=generator).to(dtype)
        return noise

    def _dream_step(
        self,
        *,
        noisy_targets: dict[str, torch.Tensor],
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        context_attention_mask: torch.Tensor,
        video_seq_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """One Dream denoising step against a cached Video K/V.

        Returns the predicted per-modality velocity, the advanced Dream tokens and
        the per-layer Dream K/V -- the interface the action expert will read.
        """
        # A regression Dream has no noisy target to encode: its tokens are the
        # learnable queries alone, exactly as in DreamFastWAM. Calling
        # encode_targets unconditionally raised on every rank of a six-run sweep.
        latent = (
            self.dream_expert.encode_targets(noisy_targets)
            if noisy_targets is not None
            else None
        )
        # Only pass the generative kwargs when there is something to pass: a
        # plain DreamQueryExpert does not accept them at all, and relying on the
        # expert always having been promoted is a dependency worth not having.
        dream_kwargs = dict(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            context=context,
            context_mask=context_mask,
        )
        if latent is not None or timestep is not None:
            dream_kwargs.update(noisy_latent=latent, timestep=timestep)
        dream_pre = self.dream_expert.pre_dit(**dream_kwargs)
        out = self.mot.forward_dream_with_video_cache(
            dream_tokens=dream_pre["tokens"],
            dream_freqs=dream_pre["freqs"],
            dream_t_mod=dream_pre["t_mod"],
            dream_context_payload={
                "context": dream_pre["context"],
                "mask": dream_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            context_attention_mask=context_attention_mask,
            video_seq_len=video_seq_len,
        )
        prediction = dream_expert.post_dit(out["tokens"], dream_pre)
        return {
            "prediction": prediction,
            "tokens": out["tokens"],
            "dream_kv": out["dream_kv"],
            "pre_state": dream_pre,
        }

    def _run_dream_rollout(
        self,
        *,
        num_steps: int,
        scheduler: WanContinuousFlowMatchScheduler,
        initial_targets: dict[str, torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        context_attention_mask: torch.Tensor,
        video_seq_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Iterate `_dream_step`; the K/V of the final step is the interface."""
        timesteps, deltas = scheduler.build_inference_schedule(
            num_inference_steps=int(num_steps), device=device, dtype=dtype
        )
        current = dict(initial_targets)
        result: dict[str, Any] = {}
        for index in range(int(num_steps)):
            timestep = timesteps[index].reshape(1).expand(batch_size)
            result = self._dream_step(
                noisy_targets=current,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                context_attention_mask=context_attention_mask,
                video_seq_len=video_seq_len,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            current = {
                name: scheduler.step(
                    model_output=result["prediction"][name].to(value.dtype),
                    delta=deltas[index],
                    sample=value,
                )
                for name, value in current.items()
                if name in result["prediction"]
            }
        result["targets"] = current
        return result

    # ------------------------------------------------------------ split prefill
    def _video_prefill(
        self,
        *,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        action_seq_len: int,
    ) -> dict[str, Any]:
        """Run the frozen Video expert once and cache its per-layer K/V."""
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
        video_seq_len = int(video_pre["tokens"].shape[1])
        dream_seq_len = int(self.dream_expert.num_dream_tokens)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            dream_seq_len=dream_seq_len,
            action_seq_len=int(action_seq_len),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        return {
            "video_kv": video_kv,
            "attention_mask": attention_mask,
            "video_seq_len": video_seq_len,
            "dream_seq_len": dream_seq_len,
        }

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
        """Inference-time prefill.

        Keeps the parent's return contract so ``DreamFastWAM.infer_action`` and
        every evaluation script work unchanged; only the way the cache is
        produced differs.  In generative mode the Video expert runs once and the
        Dream expert denoises for ``dream_inference_steps`` steps against that
        cache -- which, after interface distillation, is a single step.
        """
        if not self.dream_expert.generative_enabled:
            return super()._prefill_video_dream_cache(
                first_frame_latents=first_frame_latents,
                action_seq_len=action_seq_len,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                return_dream=return_dream,
            )

        batch_size = int(first_frame_latents.shape[0])
        prefill = self._video_prefill(
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            action_seq_len=action_seq_len,
        )
        context_seq_len = prefill["video_seq_len"] + prefill["dream_seq_len"]
        rollout = self._run_dream_rollout(
            num_steps=self.dream_inference_steps,
            scheduler=self.infer_dream_scheduler,
            initial_targets=self._dream_noise_like_targets(
                batch_size=batch_size,
                device=first_frame_latents.device,
                dtype=first_frame_latents.dtype,
            ),
            context=context,
            context_mask=context_mask,
            video_kv_cache=prefill["video_kv"],
            context_attention_mask=prefill["attention_mask"][:context_seq_len, :context_seq_len],
            video_seq_len=prefill["video_seq_len"],
            batch_size=batch_size,
            device=first_frame_latents.device,
            dtype=first_frame_latents.dtype,
        )
        return {
            "kv_cache": self.mot.merge_context_cache(prefill["video_kv"], rollout["dream_kv"]),
            "attention_mask": prefill["attention_mask"],
            "video_seq_len": prefill["video_seq_len"],
            "dream_seq_len": prefill["dream_seq_len"],
            "dream_predictions": rollout["targets"] if return_dream else None,
        }

    def _prepare_action_cache(self, cache, *, context, context_mask, proprio=None, training=False):
        """Compute one stable [B,16] gate vector before any Action denoising.

        Context = last Video layer's pooled V + pooled language + current raw
        proprio. The appended proprio token is excluded from language pooling.
        Training calls this outside the frozen Video no_grad block.
        """
        router = getattr(self.mot, "feature_router", None)
        if router is None:
            return cache
        routing_context = self.routing_context(cache, context, context_mask, proprio)
        gates = router(routing_context, training=training)
        if not training:
            if self.routing_eval_policy == "zero":
                gates = torch.zeros_like(gates)
            elif self.routing_eval_policy == "calibrated_mean":
                gates = self.routing_eval_gates.to(gates).reshape(1, 16).expand_as(gates)
        return router.gate_cache(cache, gates)

    def routing_context(self, cache, context, context_mask, proprio=None):
        """Stable control context, also reusable by offline Router calibration."""
        video = cache["kv_cache"][-1]["v"][:, :cache["video_seq_len"]].float().mean(dim=1)
        language, mask = context, context_mask
        if proprio is not None and self.proprio_dim is not None:
            language, mask = context[:, :-1], context_mask[:, :-1]
        mask = mask.to(dtype=torch.float32).unsqueeze(-1)
        language = (language.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        parts = [video, language]
        if self.proprio_dim is not None:
            state = video.new_zeros((video.shape[0], self.proprio_dim)) if proprio is None else proprio
            if tuple(state.shape) != (video.shape[0], self.proprio_dim):
                raise ValueError("Router proprio must be the current state [B,proprio_dim].")
            parts.append(state.to(device=video.device, dtype=torch.float32))
        return torch.cat(parts, dim=-1)

    def set_routing_evaluation(self, policy: str = "native", gates=None):
        if policy not in {"native", "zero", "calibrated_mean"}:
            raise ValueError("Unknown evaluation routing policy.")
        if self.mot.feature_router is None:
            router = self.mot.router
            if router is None or router.config.mode != "learned":
                raise ValueError("Routing intervention requires a semantic Router.")
            if policy != "native":
                raise ValueError("zero/calibrated_mean interventions require the legacy signed QK Router.")
        if policy == "calibrated_mean":
            gates = torch.as_tensor(gates, dtype=torch.float32)
            if gates.shape != (16,) or not torch.isfinite(gates).all() or not ((gates >= -1) & (gates <= 1)).all():
                raise ValueError("Calibration must contain 16 finite gates in [-1,1].")
        self.routing_eval_policy, self.routing_eval_gates = policy, gates

    def routing_gradient_metrics(self):
        # Local last-microbatch norms, before distributed averaging/clipping.
        # Hooks work with DeepSpeed too, whose optimizer may already have cleared .grad.
        groups = {"prior": [], "mlp": []}
        for name, norm in self._routing_gradient_norms.items():
            groups["prior" if name == "gate_prior" else "mlp"].append(norm.square())
        self._routing_gradient_norms.clear()
        return {f"router_{name}_grad_local_microbatch_norm": float(torch.stack(norms).sum().sqrt())
                for name, norms in groups.items() if norms}

    def dense_context(self, inputs, *, decode: bool = False):
        """Differentiable dense prefill. Frozen Video never constructs an autograd graph."""
        if self.dream_expert.generative_enabled:
            raise ValueError("dense_context cannot be used with a generative Dream.")
        current = inputs["first_frame_latents"]
        if current is None:
            current = inputs["input_latents"][:, :, :1]
        with torch.no_grad():
            prefill = self._video_prefill(
                first_frame_latents=current, context=inputs["context"],
                context_mask=inputs["context_mask"],
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                action_seq_len=inputs["action"].shape[1],
            )
        length = prefill["video_seq_len"] + prefill["dream_seq_len"]
        with torch.no_grad() if self.training_mode == "router_only" else nullcontext():
            pre = self.dream_expert.pre_dit(
                batch_size=current.shape[0], device=current.device, dtype=current.dtype,
                context=inputs["context"], context_mask=inputs["context_mask"],
            )
            out = self.mot.forward_dream_with_video_cache(
                dream_tokens=pre["tokens"], dream_freqs=pre["freqs"], dream_t_mod=pre["t_mod"],
                dream_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
                video_kv_cache=prefill["video_kv"],
                context_attention_mask=prefill["attention_mask"][:length, :length],
                video_seq_len=prefill["video_seq_len"],
            )
            predictions = self.dream_expert.post_dit(out["tokens"], pre) if decode else None
        cache = {key: prefill[key] for key in ("video_seq_len", "dream_seq_len", "attention_mask")}
        cache["kv_cache"] = self.mot.merge_context_cache(prefill["video_kv"], out["dream_kv"])
        return cache, predictions

    def dense_action_loss(self, inputs, cache, *, noise: torch.Tensor, timestep: torch.Tensor):
        """Return scheduler-weighted per-sample loss; Action input gradients remain enabled."""
        action = inputs["action"]
        scheduler = self.train_action_scheduler
        noisy = scheduler.add_noise(action, noise, timestep)
        target = scheduler.training_target(action, noise, timestep)
        pre = self.action_expert.pre_dit(action_tokens=noisy, timestep=timestep,
            context=inputs["context"], context_mask=inputs["context_mask"])
        tokens = self.mot.forward_action_with_context_cache(
            action_tokens=pre["tokens"], action_freqs=pre["freqs"], action_t_mod=pre["t_mod"],
            action_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
            context_kv_cache=cache["kv_cache"], attention_mask=cache["attention_mask"],
            video_seq_len=cache["video_seq_len"], dream_seq_len=cache["dream_seq_len"],
        )
        pred = self.action_expert.post_dit(tokens, pre)
        error = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)
        valid = inputs.get("action_is_pad")
        if valid is None:
            per_sample = error.mean(dim=1)
        else:
            valid = (~valid).to(error)
            per_sample = (error * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return per_sample * scheduler.training_weight(timestep).to(per_sample)

    def _dense_training_loss(self, sample, tiled=False):
        if self.loss_lambda_video != 0:
            raise ValueError("Dense cached training requires lambda_video=0.")
        inputs = self.build_inputs(sample, tiled=tiled)
        train_features = self.training_mode != "router_only"
        cache, predictions = self.dense_context(inputs, decode=train_features)
        cache = self._prepare_action_cache(cache, context=inputs["context"],
            context_mask=inputs["context_mask"], training=True,
            proprio=sample["proprio"][:, 0] if sample.get("proprio") is not None else None)
        action = inputs["action"]
        if "routing_sample_id" in sample:
            from fastwam.utils.routing_experiment import paired_action_noise
            noise, timestep = paired_action_noise(action, sample["routing_sample_id"],
                scheduler=self.train_action_scheduler, seed=int(sample["routing_noise_seed"][0])
                + self._routing_global_step * 10_007)
        else:
            noise = self._sample_action_noise(action, use_correlated_noise=self.use_correlated_noise_train)
            timestep = self.train_action_scheduler.sample_training_t(
                batch_size=action.shape[0], device=action.device, dtype=action.dtype)
        loss_action = self.dense_action_loss(inputs, cache, noise=noise, timestep=timestep).mean()
        total = self.loss_lambda_action * loss_action
        metrics = {"action_loss": float(loss_action.detach()), "loss_action": float(total.detach())}
        if train_features:
            self._assert_training_dream_modalities_match(inputs["dream_targets"])
            feature_loss, parts = self._compute_dream_loss(predictions, inputs["dream_targets"],
                future_valid_mask=inputs.get("future_valid_mask"),
                modality_valid_masks=inputs.get("modality_valid_masks"), future_offsets=inputs.get("future_offsets"))
            total = total + self.loss_lambda_dream * feature_loss
            metrics.update(feature_loss=float(feature_loss.detach()),
                           loss_dream=float((self.loss_lambda_dream * feature_loss).detach()))
            for name in self.dream_expert.modalities:
                metrics["loss_" + name] = self.loss_lambda_dream * getattr(self, "loss_lambda_" + name) * float(parts["loss_" + name].detach())
        return self._add_router_terms(total, metrics, cache.get("group_gates"))

    # ------------------------------------------------------------------- losses
    def _generative_dream_loss(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        *,
        future_valid_mask: Optional[torch.Tensor],
        modality_valid_masks: Optional[dict[str, torch.Tensor]],
        sample_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Flow-matching MSE per modality, masked exactly like the dense loss.

        The dense loss uses modality-specific objectives (BCE for the dynamic
        mask, SiLog for depth, cosine for DINO/SAM).  Those are objectives on
        the *value*; here the network regresses a velocity, for which a single
        squared error is the right and only consistent choice.
        """
        modality_lambdas = {
            "dyn": self.loss_lambda_dyn,
            "depth": self.loss_lambda_depth,
            "dino": self.loss_lambda_dino,
            "sam": self.loss_lambda_sam,
        }
        device = next(iter(prediction.values())).device
        total = torch.zeros((), device=device, dtype=torch.float32)
        parts: dict[str, torch.Tensor] = {}
        for name in self.dream_expert.modalities:
            if name not in prediction:
                continue
            pred = prediction[name].float()
            goal = target[name].float()
            if pred.shape != goal.shape:
                raise ValueError(
                    f"{name} velocity shape mismatch: {tuple(pred.shape)} vs {tuple(goal.shape)}."
                )
            per_offset = F.mse_loss(pred, goal, reduction="none")
            per_offset = per_offset.flatten(2).mean(dim=2)  # [B, O]
            # The scheduler weight belongs to this sample's noise timestep.
            # Averaging weights before the loss introduces cross-sample terms.
            if sample_weights is not None:
                weights = sample_weights.to(device=per_offset.device, dtype=per_offset.dtype)
                if weights.numel() != per_offset.shape[0]:
                    raise ValueError("Dream sample_weights must contain one weight per sample.")
                per_offset = per_offset * weights.reshape(-1, 1)
            mask = None
            if modality_valid_masks is not None and name in modality_valid_masks:
                mask = modality_valid_masks[name].to(device=per_offset.device, dtype=per_offset.dtype)
            elif future_valid_mask is not None:
                mask = future_valid_mask.to(device=per_offset.device, dtype=per_offset.dtype)
            if mask is not None:
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0).expand_as(per_offset)
                value = (per_offset * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                value = per_offset.mean()
            parts[f"loss_{name}"] = value.detach()
            total = total + float(modality_lambdas.get(name, 1.0)) * value
        return total, parts

    def training_loss(self, sample, tiled: bool = False):
        self._refresh_progress()
        if (not self.dream_expert.generative_enabled
                and (self.training_mode == "dense_joint" or self.mot.feature_router is not None)):
            return self._dense_training_loss(sample, tiled=tiled)
        if not self.uses_split_path:
            loss, metrics = super().training_loss(sample, tiled=tiled)
            loss, metrics = self._add_router_terms(loss, metrics)
            return loss, metrics
        return self._split_training_loss(sample, tiled=tiled)

    def _add_router_terms(self, loss, metrics: dict[str, float], group_gates=None):
        metrics = dict(metrics)
        feature_router = getattr(self.mot, "feature_router", None)
        if feature_router is not None:
            if group_gates is None:
                raise RuntimeError("Semantic routing loss requires the gates used by Action.")
            gate_loss = feature_router.gate_loss(group_gates, training=True)
            weighted = feature_router.config.router_gate_loss_weight * gate_loss
            metrics.update(gate_loss=float(gate_loss.detach()),
                           mean_gate=float(group_gates.detach().mean()),
                           loss_router_gate=float(weighted.detach()))
            activation = feature_router.activations(group_gates.detach()).mean(dim=0)
            metrics.update({name: float(value) for name, value in
                            zip(feature_router.activation_names, activation)})
            for index in range(16):
                values = group_gates.detach().float()[:, index]
                metrics[f"gate_{index:02d}_mean"] = float(values.mean())
                metrics[f"gate_{index:02d}_variance"] = float(values.var(unbiased=False))
            return loss + weighted, metrics
        router = self.mot.router
        if router is None:
            return loss, metrics
        gates = self.mot.last_group_gates
        if router.config.mode == "learned":
            if len(gates) != self.mot.num_layers:
                raise RuntimeError(
                    "Learned routing requires group gates from every Action layer; "
                    f"collected {len(gates)} of {self.mot.num_layers}."
                )
        # Gates are learned exclusively through Action loss. These are diagnostic
        # metrics, with no budget, sparsity, entropy or gate-supervision objective.
        metrics["router_layers"] = len(gates)
        metrics["router_strength"] = router.current_strength()
        metrics.update(router.scalar_metrics())
        return loss, metrics

    @torch.no_grad()
    def infer_action(self, *args, **kwargs):
        """Export the final Action denoising forward, retaining every layer's gates.

        group_gates is the layer mean for existing rollout consumers;
        group_gates_per_layer is [B,L,Ng] for full trajectory/layer heatmaps.
        """
        self.mot._reset_router_records()
        output = super().infer_action(*args, **kwargs)
        router = self.mot.router
        if router is not None and router.config.mode == "learned" and self.mot.last_group_gates:
            per_layer = torch.stack(self.mot.last_group_gates, dim=1).detach().float().cpu()
            groups = per_layer.mean(dim=1)
            output.update(group_gates=groups, group_gates_per_layer=per_layer,
                          group_mapping=router.group_mapping, gate_kind="attention_prior",
                          gate_denoising_step="last", gate_layer_reduction="mean")
            for modality, name in (("dino", "dino"), ("dyn", "tracker"),
                                   ("sam", "sam"), ("depth", "depth")):
                indices = [g["group_id"] for g in router.group_mapping if g["modality"] == modality]
                if indices:
                    output[f"{name}_activation"] = groups[:, indices].mean(dim=-1)
        return output

    def _split_training_loss(self, sample, tiled: bool = False):
        """Train through the deployment computation: Video once, Dream, Action cached."""
        if self.online_targets is not None:
            # Skip DreamFastWAM.build_inputs: it insists on `sample["dream_targets"]`,
            # which only exists when precomputed extras are on disk. With online
            # targets the dataset does not need to supply them at all, so
            # data.train.dream_target.enabled can stay false.
            inputs = FastWAM.build_inputs(self, sample, tiled=tiled)
        else:
            inputs = self.build_inputs(sample, tiled=tiled)
        if self.loss_lambda_video > 0.0:
            raise ValueError(
                "The split path does not denoise future video frames; set "
                "model.loss.lambda_video=0.0 for generative/interface-distilled runs."
            )
        input_latents = inputs["input_latents"]
        batch_size = int(input_latents.shape[0])
        device = input_latents.device
        dtype = input_latents.dtype
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        if self.online_targets is not None:
            # The future frames the offsets point at are already in this clip:
            # video_sample_indices == [0, 4, ..., 32] and future_offsets are
            # multiples of action_video_freq_ratio, so no extra I/O is needed.
            online = self.online_targets(
                sample["video"],
                future_offsets=list(self.dream_expert.future_offsets),
                action_video_freq_ratio=int(self.online_action_video_freq_ratio),
                image_is_pad=inputs.get("image_is_pad", None),
            )
            dream_targets = self._dream_target_reference(online)
            future_valid_mask = online["future_valid_mask"]
            modality_valid_masks = {
                name: online[f"{name}_valid_mask"]
                for name in self.dream_expert.modalities
                if f"{name}_valid_mask" in online
            }
        else:
            dream_targets = self._dream_target_reference(inputs["dream_targets"])
            future_valid_mask = inputs.get("future_valid_mask", None)
            modality_valid_masks = inputs.get("modality_valid_masks", None)
        first_frame_latents = inputs["first_frame_latents"]
        if first_frame_latents is None:
            first_frame_latents = input_latents[:, :, 0:1]

        # --- Video prefill (frozen; no gradient path, no optimizer state) ---
        with torch.no_grad():
            prefill = self._video_prefill(
                first_frame_latents=first_frame_latents,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                action_seq_len=int(action.shape[1]),
            )
        context_seq_len = prefill["video_seq_len"] + prefill["dream_seq_len"]
        context_attention_mask = prefill["attention_mask"][:context_seq_len, :context_seq_len]

        # --- Dream ---
        clean_targets = {name: value.to(dtype) for name, value in dream_targets.items()}
        generative = bool(self.dream_expert.generative_enabled)
        if generative:
            # Flow matching in target space. NOTE this is only well posed when the
            # Dream bottleneck can carry the denoising state: the velocity target
            # `noise - x0` is per-token i.i.d., while the encoder compresses e.g.
            # DINO's 256x768 per camera into 9x384 (57x). Recovering per-token
            # noise from that is information-theoretically impossible, and a run
            # with this on sits at loss_dream ~= 1 + var(y) -- the conditional
            # mean -- which is what jobs E1-E6 measured.
            dream_noise = self._sample_dream_noise(clean_targets)
            timestep_dream = self.train_dream_scheduler.sample_training_t(
                batch_size=batch_size, device=device, dtype=dtype
            )
            noisy_targets = {
                name: self.train_dream_scheduler.add_noise(value, dream_noise[name], timestep_dream)
                for name, value in clean_targets.items()
            }
            supervision = {
                name: self.train_dream_scheduler.training_target(
                    value, dream_noise[name], timestep_dream
                )
                for name, value in clean_targets.items()
            }
        else:
            # Regression Dream: predict the future target itself. Well posed at
            # any bottleneck width, and what DreamFastWAM validates.
            noisy_targets = None
            timestep_dream = None
            supervision = clean_targets
        dream_out = self._dream_step(
            noisy_targets=noisy_targets,
            timestep=timestep_dream,
            context=context,
            context_mask=context_mask,
            video_kv_cache=prefill["video_kv"],
            context_attention_mask=context_attention_mask,
            video_seq_len=prefill["video_seq_len"],
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        if generative:
            loss_dream, dream_parts = self._generative_dream_loss(
                dream_out["prediction"],
                supervision,
                future_valid_mask=future_valid_mask,
                modality_valid_masks=modality_valid_masks,
            )
            dream_weight = self.train_dream_scheduler.training_weight(timestep_dream).mean()
            loss_dream = loss_dream * dream_weight.to(loss_dream.dtype)
        else:
            # Reuse the parent's per-modality objectives (smooth-L1 for depth,
            # 1 - cosine for DINO/SAM); a single MSE is only right for velocity.
            loss_dream, dream_parts = self._compute_dream_loss(
                dream_out["prediction"],
                dict(supervision),
                future_valid_mask=future_valid_mask,
                modality_valid_masks=modality_valid_masks,
            )

        # --- Action: denoise against the merged cache ---
        noise_action = self._sample_action_noise(
            action, use_correlated_noise=self.use_correlated_noise_train
        )
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_cache = self._prepare_action_cache(
            {"kv_cache": self.mot.merge_context_cache(prefill["video_kv"], dream_out["dream_kv"]),
             "video_seq_len": prefill["video_seq_len"], "dream_seq_len": prefill["dream_seq_len"]},
            context=context, context_mask=context_mask,
            proprio=sample["proprio"][:, 0] if sample.get("proprio") is not None else None,
            training=True,
        )
        action_tokens = self.mot.forward_action_with_context_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            context_kv_cache=action_cache["kv_cache"],
            attention_mask=prefill["attention_mask"],
            video_seq_len=prefill["video_seq_len"],
            dream_seq_len=prefill["dream_seq_len"],
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)

        token_loss = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is None:
            per_sample = token_loss.mean(dim=1)
        else:
            valid = (~action_is_pad).to(device=token_loss.device, dtype=token_loss.dtype)
            per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            device=per_sample.device, dtype=per_sample.dtype
        )
        loss_action = (per_sample * action_weight).mean()

        loss_total = self.loss_lambda_action * loss_action + self.loss_lambda_dream * loss_dream
        metrics = {
            "action_loss": float(loss_action.detach()),
            "feature_loss": float(loss_dream.detach()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_dream": self.loss_lambda_dream * float(loss_dream.detach().item()),
            "dream_timestep_mean": (
                float(timestep_dream.detach().float().mean().item())
                if timestep_dream is not None
                else -1.0
            ),
        }
        for name, value in dream_parts.items():
            # Only the loss terms are scaled by lambda_dream; the parent also
            # returns validity ratios and counts, which must be logged as-is.
            scale = self.loss_lambda_dream if name.startswith("loss_") else 1.0
            metrics[name] = scale * float(value.item())

        # --- Interface distillation ---
        if self.distiller is not None:
            loss_iface, iface_metrics = self._interface_distillation_loss(
                student_kv=dream_out["dream_kv"],
                context=context,
                context_mask=context_mask,
                video_kv_cache=prefill["video_kv"],
                context_attention_mask=context_attention_mask,
                video_seq_len=prefill["video_seq_len"],
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            loss_total = loss_total + self.distiller.current_weight() * loss_iface
            metrics.update(iface_metrics)

        loss_total, metrics = self._add_router_terms(loss_total, metrics, action_cache.get("group_gates"))
        return loss_total, metrics

    def _interface_distillation_loss(
        self,
        *,
        student_kv: list[dict[str, torch.Tensor]],
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        context_attention_mask: torch.Tensor,
        video_seq_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert self.distiller is not None
        with torch.no_grad():
            with self.distiller.use_teacher_dream(self.mot):
                teacher = self._run_dream_rollout(
                    num_steps=self.distill_config.teacher_steps,
                    scheduler=self.infer_dream_scheduler,
                    initial_targets=self._dream_noise_like_targets(
                        batch_size=batch_size, device=device, dtype=dtype
                    ),
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    context_attention_mask=context_attention_mask,
                    video_seq_len=video_seq_len,
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                )

        keep_mask = None
        if self.distill_config.route_aware and self.mot.router is not None:
            gates = getattr(self.mot, "last_gates", [])
            if gates:
                keep_mask = (torch.stack(gates, dim=0).mean(dim=0) > self.router_config.gate_threshold)

        loss, parts = self.distiller.kv_loss(
            student_kv=student_kv,
            teacher_kv=teacher["dream_kv"],
            keep_mask=keep_mask,
        )
        metrics = {
            "loss_interface": float(loss.detach().item()),
            "interface_weight": float(self.distiller.current_weight()),
        }
        for key, value in parts.items():
            metrics[key] = float(value.item())
        return loss, metrics
