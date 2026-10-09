"""Action-side routing over the imagination interface.

`DynamicFeatureRouter` implements full/static/dynamic semantic QK gating:
one [B,16] vector from stable control context, shared across layers and noise
steps. `ImaginationRouter` uses the separate `routing_mode=null` path below.

The action expert in a DreamFastWAM MoT reads the world branch only through the
per-layer keys/values of the Video and Dream tokens.  `ImaginationRouter` lets
the action expert decide, per layer and per sample, *which* of those Dream keys
it wants to read, instead of attending to all of them with a fixed mask.

Three modes are supported, all sharing one interface so they can be swapped from
config:

``none``
    No routing.  The caller is expected to take the dense fast path.

``threshold``
    The parameter-free policy already shipped in
    ``models/wan22/action_dream_threshold``: score a Dream key by its dense
    attention mass normalised by the uniform attention level, then keep it when
    the score clears ``alpha``.  ``alpha == 1`` therefore means "attended more
    than uniformly".  Reimplemented here (rather than imported) only so that the
    three modes share one return contract; the arithmetic is identical and
    ``tests/test_routed_wam.py`` asserts equality against the original.

``learned``
    One sigmoid gate per layer and Dream semantic group. ``gate_type=va`` uses
    current observed Video keys and Action queries, without Dream features.
    ``gate_type=da`` uses predicted Dream keys and Action queries instead.
    ``gate_type=static`` learns input-independent logits for each layer/group.
    The resulting gate is
    injected into the attention logits additively as ``log(gate)``, which is
    exactly equivalent to scaling the unnormalised attention weight by ``gate``:
    a gate of 1 leaves the dense distribution untouched, and a gate of 0
    reproduces a hard mask.  That equivalence is what makes the dense model a
    strict special case of the routed one.

The router is deliberately **deterministic**.  Mixed attention runs inside
``torch.utils.checkpoint``, which re-executes the wrapped function during
backward; anything stochastic (Gumbel noise, dropout) would draw different
values on the recompute and silently corrupt gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger


logger = get_logger(__name__)

ROUTER_MODES = ("none", "threshold", "learned")
GATE_TYPES = ("va", "da", "static")

#: How Dream tokens are bucketed for the budget bias, logging and analysis.
#: The `_camera` variants split each block into its primary and wrist halves,
#: which is what the stage-dependent-perception study needs.
GROUP_GRANULARITIES = (
    "none",
    "modality",
    "modality_horizon",
    "modality_camera",
    "modality_horizon_camera",
)


@dataclass(frozen=True)
class RouterConfig:
    """Configuration for :class:`ImaginationRouter`."""

    mode: str = "none"
    # Semantic QK routing. None selects ImaginationRouter above.
    routing_mode: Optional[str] = None
    router_enabled: bool = True
    router_warmup_steps: int = 0
    router_hidden_dim: int = 128
    router_gate_loss_weight: float = 0.0
    gate_init_probability: float = 0.5  # Deprecated; accepted for old configs. QK gates always start at zero.
    # --- threshold mode ---
    alpha: float = 1.0
    # --- learned mode ---
    gate_type: str = "va"
    rank: int = 64
    temperature: float = 1.0
    bias_init: float = 4.0
    group_granularity: str = "modality_horizon"
    # --- shared ---
    min_keep_tokens: int = 0
    target_keep_ratio: float = 0.25  # Compatibility only; no budget objective.
    lambda_budget: float = 0.0  # Compatibility only; learned routing requires zero.
    gate_threshold: float = 1.0e-3
    hard_prune_at_inference: bool = True
    warmup_ratio: float = 0.1
    log_statistics: bool = True
    debug_force_gate: Optional[float] = None
    layers: Optional[list[int]] = field(default=None)

    @classmethod
    def from_dict(cls, value: Optional[dict[str, Any]]) -> "RouterConfig":
        cfg = cls(**({} if value is None else dict(value)))
        if cfg.routing_mode not in (None, "full", "static", "dynamic"):
            raise ValueError("routing_mode must be full, static, dynamic, or null (legacy).")
        if cfg.routing_mode is not None and cfg.mode != "none":
            raise ValueError("Semantic routing requires router.mode=none; legacy routing cannot also run.")
        if cfg.router_warmup_steps < 0 or cfg.router_hidden_dim <= 0 or cfg.router_gate_loss_weight < 0:
            raise ValueError("Invalid router warmup, hidden dimension, or gate loss weight.")
        if not 0 < cfg.gate_init_probability < 1:
            raise ValueError("gate_init_probability must be strictly between zero and one.")
        if cfg.mode not in ROUTER_MODES:
            raise ValueError(f"router.mode must be one of {ROUTER_MODES}, got {cfg.mode!r}.")
        if cfg.gate_type not in GATE_TYPES:
            raise ValueError(f"router.gate_type must be one of {GATE_TYPES}, got {cfg.gate_type!r}.")
        if cfg.alpha < 0:
            raise ValueError(f"router.alpha must be >= 0, got {cfg.alpha}.")
        if cfg.rank <= 0:
            raise ValueError(f"router.rank must be > 0, got {cfg.rank}.")
        if cfg.temperature <= 0:
            raise ValueError(f"router.temperature must be > 0, got {cfg.temperature}.")
        if not 0.0 <= cfg.target_keep_ratio <= 1.0:
            raise ValueError(
                f"router.target_keep_ratio must be in [0,1], got {cfg.target_keep_ratio}."
            )
        if cfg.lambda_budget < 0:
            raise ValueError(f"router.lambda_budget must be >= 0, got {cfg.lambda_budget}.")
        if cfg.mode == "learned" and cfg.lambda_budget != 0:
            raise ValueError("Learned group routing uses Action loss only; lambda_budget must be 0.")
        if not 0.0 <= cfg.warmup_ratio <= 1.0:
            raise ValueError(f"router.warmup_ratio must be in [0,1], got {cfg.warmup_ratio}.")
        if cfg.min_keep_tokens < 0:
            raise ValueError(f"router.min_keep_tokens must be >= 0, got {cfg.min_keep_tokens}.")
        if cfg.group_granularity not in GROUP_GRANULARITIES:
            raise ValueError(
                "router.group_granularity must be one of "
                f"{GROUP_GRANULARITIES}, got {cfg.group_granularity!r}."
            )
        if cfg.debug_force_gate is not None and not 0.0 <= float(cfg.debug_force_gate) <= 1.0:
            raise ValueError("router.debug_force_gate must be in [0,1] when set.")
        return cfg

    @property
    def enabled(self) -> bool:
        return self.mode != "none"


GROUP_GRANULARITIES = ("none", "modality", "modality_horizon", "modality_horizon_view")


def build_group_ids(
    *,
    modalities: list[str],
    num_future_offsets: int,
    tokens_per_modality: dict[str, int],
    granularity: str,
    camera_token_split: Optional[tuple[int, int]] = None,
) -> tuple[torch.Tensor, list[str]]:
    """Map each Dream token to an interpretable group id.

    Dream tokens are laid out modality-major, then offset-major, and within one
    (modality, offset) block the first ``camera_token_split[0]`` tokens belong to
    the primary camera and the rest to the wrist camera -- the same convention
    ``DenseDreamDecoder.forward_two_view`` decodes with. The grouping is
    therefore a pure function of the layout and needs no runtime information.

    The camera granularities exist for the stage-dependent-perception analysis:
    the hypothesis that a policy leans on global semantics (DINO/SAM, primary
    view) early in an episode and on local geometry (depth, wrist view) near
    contact is only testable if the wrist tokens are a separate group.

    Returns:
        ids: LongTensor [num_dream_tokens] of group indices.
        names: human-readable name per group index, used for logging and figures.
    """
    if granularity not in GROUP_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {GROUP_GRANULARITIES}, got {granularity!r}."
        )
    with_camera = granularity.endswith("_camera")
    with_horizon = "horizon" in granularity
    if with_camera and camera_token_split is None:
        raise ValueError(
            f"granularity={granularity!r} needs dream_query.camera_token_split; "
            "without it the primary and wrist tokens are indistinguishable."
        )

    cameras = ["primary", "wrist"] if with_camera else [None]

    def group_name(modality: str, offset_index: int, camera: Optional[str]) -> str:
        if granularity == "none":
            return "all"
        name = modality
        if with_horizon:
            name = f"{name}@t{offset_index}"
        if camera is not None:
            name = f"{name}/{camera}"
        return name

    names: list[str] = []
    index_by_name: dict[str, int] = {}
    for modality in modalities:
        for offset_index in range(num_future_offsets if with_horizon else 1):
            for camera in cameras:
                name = group_name(modality, offset_index, camera)
                if name not in index_by_name:
                    index_by_name[name] = len(names)
                    names.append(name)

    ids: list[int] = []
    for modality in modalities:
        per_offset = int(tokens_per_modality[modality])
        if with_camera:
            primary_tokens, wrist_tokens = camera_token_split
            if primary_tokens + wrist_tokens != per_offset:
                raise ValueError(
                    f"camera_token_split {list(camera_token_split)} must sum to "
                    f"n_{modality}={per_offset}."
                )
        for offset_index in range(num_future_offsets):
            horizon_index = offset_index if with_horizon else 0
            if with_camera:
                primary_tokens, wrist_tokens = camera_token_split
                ids.extend(
                    [index_by_name[group_name(modality, horizon_index, "primary")]] * primary_tokens
                )
                ids.extend(
                    [index_by_name[group_name(modality, horizon_index, "wrist")]] * wrist_tokens
                )
            else:
                ids.extend([index_by_name[group_name(modality, horizon_index, None)]] * per_offset)
    return torch.tensor(ids, dtype=torch.long), names


class ImaginationRouter(nn.Module):
    """Per-layer gate over the Action -> Dream attention edges."""

    def __init__(
        self,
        *,
        config: RouterConfig,
        num_layers: int,
        inner_dim: int,
        num_dream_tokens: int,
        group_ids: torch.Tensor,
        group_names: list[str],
        future_offsets: Optional[list[int]] = None,
    ):
        super().__init__()
        self.config = config
        self.num_layers = int(num_layers)
        self.inner_dim = int(inner_dim)
        self.num_dream_tokens = int(num_dream_tokens)
        self.group_names = list(group_names)

        if group_ids.numel() != self.num_dream_tokens:
            raise ValueError(
                f"group_ids has {group_ids.numel()} entries but there are "
                f"{self.num_dream_tokens} dream tokens."
            )
        self.register_buffer("group_ids", group_ids.clone(), persistent=False)
        if (not self.group_names or group_ids.ndim != 1 or group_ids.dtype != torch.long
                or bool((group_ids < 0).any()) or bool((group_ids >= len(self.group_names)).any())):
            raise ValueError("group_ids must be a 1D LongTensor indexing group_names.")
        counts = torch.bincount(group_ids, minlength=len(self.group_names))
        if bool((counts == 0).any()):
            raise ValueError("Every Dream group must contain at least one token.")
        self.register_buffer("group_token_counts", counts, persistent=False)
        self.group_mapping = []
        for index, name in enumerate(self.group_names):
            parts = name.split("@")
            horizon = int(parts[1][1:]) if len(parts) > 1 else None
            positions = (group_ids == index).nonzero().flatten()
            self.group_mapping.append(dict(
                group_id=index, name=name, modality=parts[0],
                horizon_index=horizon, view=parts[2] if len(parts) > 2 else None,
                future_offset=(int(future_offsets[horizon]) if future_offsets is not None
                               and horizon is not None else horizon),
                token_start=int(positions[0]), token_stop=int(positions[-1]) + 1,
            ))

        self.active_layers = (
            set(range(self.num_layers))
            if config.layers is None
            else {int(x) for x in config.layers}
        )
        unknown = {x for x in self.active_layers if not 0 <= x < self.num_layers}
        if unknown:
            raise ValueError(f"router.layers contains out-of-range entries: {sorted(unknown)}")

        if config.gate_type not in GATE_TYPES:
            raise ValueError(f"router.gate_type must be one of {GATE_TYPES}, got {config.gate_type!r}.")
        self.action_proj = None
        self.video_proj = None
        self.dream_proj = None
        self.gate_proj = None
        self.register_parameter("static_logits", None)
        if config.mode == "learned" and config.lambda_budget != 0:
            raise ValueError("Learned group routing uses Action loss only; lambda_budget must be 0.")
        if config.mode == "learned" and config.gate_type == "static":
            self.static_logits = nn.Parameter(torch.full(
                (self.num_layers, len(self.group_names)), float(config.bias_init), dtype=torch.float32
            ))
        elif config.mode == "learned":
            rank = int(config.rank)
            self.action_proj = nn.ModuleList(
                [nn.Linear(self.inner_dim, rank, bias=False) for _ in range(self.num_layers)]
            )
            source_proj = nn.ModuleList(
                [nn.Linear(self.inner_dim, rank, bias=False) for _ in range(self.num_layers)]
            )
            # Keep VA parameter names unchanged so existing VA checkpoints load.
            if config.gate_type == "va":
                self.video_proj = source_proj
            else:
                self.dream_proj = source_proj
            self.gate_proj = nn.ModuleList(
                [nn.Linear(rank, len(self.group_names)) for _ in range(self.num_layers)]
            )
            # Begin near dense attention and learn what information to suppress.
            for module in list(self.action_proj) + list(source_proj) + list(self.gate_proj):
                nn.init.normal_(module.weight, std=1.0e-3)
            for module in self.gate_proj:
                nn.init.constant_(module.bias, float(config.bias_init))

        self._progress: float = 1.0
        self.last_statistics: list[dict[str, Any]] = []
        logger.info(
            "ImaginationRouter(mode=%s, gate_type=%s) over %d dream tokens, %d groups, layers=%s",
            config.mode,
            config.gate_type,
            self.num_dream_tokens,
            len(self.group_names),
            "all" if config.layers is None else sorted(self.active_layers),
        )

    # ------------------------------------------------------------------ utils
    def _apply(self, fn, recurse=True):
        # Keep small input-dependent logit changes visible next to bias_init=4,
        # including when DeepSpeed converts the parent model to BF16.
        def preserve_precision(tensor):
            converted = fn(tensor)
            if tensor.is_floating_point():
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted
        return super()._apply(preserve_precision, recurse=recurse)

    def set_progress(self, fraction: float) -> None:
        """Warmup fraction in [0, 1]; 0 means "behave densely"."""
        self._progress = float(min(max(fraction, 0.0), 1.0))

    def current_strength(self) -> float:
        """How strongly routing is applied right now.

        Warmup only applies while training: evaluation always uses the full
        policy, otherwise a checkpoint would evaluate differently depending on
        how far through training it was saved.
        """
        if not self.training or self.config.warmup_ratio <= 0.0:
            return 1.0
        return self._progress

    def reset_statistics(self) -> None:
        self.last_statistics = []

    # ------------------------------------------------------------------ gates
    def learned_group_gate(
        self,
        *,
        layer_idx: int,
        q_action: torch.Tensor,
        k_video_current: Optional[torch.Tensor] = None,
        k_dream: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return [B, Ng] sigmoid priors for S, VA or DA, before warmup/pruning.

        DA pools the predicted Dream keys already used by mixed attention. It
        never reads future target features; gradients flow through its summary.
        Static uses only the batch shape of Q and no input features.
        """
        if self.config.mode != "learned":
            raise ValueError("learned_group_gate requires router.mode='learned'.")
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError("Router layer index is out of range.")
        if q_action.ndim != 3 or q_action.shape[1] == 0 or q_action.shape[-1] != self.inner_dim:
            raise ValueError(f"q_action must be [B, nonempty_sequence, {self.inner_dim}].")
        if self.config.gate_type == "static":
            with torch.autocast(device_type=q_action.device.type, enabled=False):
                gate = torch.sigmoid(self.static_logits[layer_idx] / float(self.config.temperature))
            return gate.unsqueeze(0).expand(q_action.shape[0], -1)
        if self.config.gate_type == "va":
            source_name, source, projection = "k_video_current", k_video_current, self.video_proj
        else:
            source_name, source, projection = "k_dream", k_dream, self.dream_proj
        if source is None:
            raise ValueError(f"Learned {self.config.gate_type} routing requires {source_name}.")
        if source.ndim != 3 or source.shape[1] == 0 or source.shape[-1] != self.inner_dim:
            raise ValueError(f"{source_name} must be [B, nonempty_sequence, {self.inner_dim}].")
        if q_action.shape[0] != source.shape[0] or q_action.device != source.device:
            raise ValueError(f"Action and {source_name} must have the same batch size and device.")
        if self.config.gate_type == "da" and source.shape[1] != self.num_dream_tokens:
            raise ValueError("Dream key count does not match the configured token layout.")
        with torch.autocast(device_type=q_action.device.type, enabled=False):
            action_summary = q_action.float().mean(dim=1)
            source_summary = source.float().mean(dim=1)
            hidden = torch.tanh(projection[layer_idx](source_summary)
                                + self.action_proj[layer_idx](action_summary))
            logits = self.gate_proj[layer_idx](hidden)
            return torch.sigmoid(logits / float(self.config.temperature))

    @staticmethod
    def _uniform_normalised_dream_mass(
        *,
        dense_probs: torch.Tensor,
        dream_slice: slice,
        n_valid: torch.Tensor,
    ) -> torch.Tensor:
        """The shipped threshold score: mean over heads, max over action queries.

        ``dense_probs`` is [B, H, Sa, S]; multiplying by the number of valid keys
        turns "probability" into "multiples of the uniform attention level", so
        the threshold is comparable across layers and sequence lengths.
        """
        dense_dream = dense_probs[..., dream_slice]
        normalised = dense_dream * n_valid[:, None, :, None]
        return normalised.mean(dim=1).amax(dim=1)

    def _apply_min_keep(self, keep: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        minimum = min(int(self.config.min_keep_tokens), int(score.shape[-1]))
        if minimum <= 0:
            return keep
        needed = (minimum - keep.sum(dim=-1)).clamp(min=0)
        if not bool((needed > 0).any()):
            return keep
        order = torch.argsort(score, dim=-1, descending=True, stable=True)
        ranks = torch.arange(score.shape[-1], device=score.device).view(1, -1)
        additions_in_rank_order = ranks < needed.unsqueeze(-1)
        additions = torch.zeros_like(keep).scatter(1, order, additions_in_rank_order)
        return keep | additions

    # ---------------------------------------------------------------- forward
    def _apply_group_min_keep(self, keep: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """Satisfy min_keep_tokens by adding whole groups, including uneven groups."""
        minimum = min(int(self.config.min_keep_tokens), self.num_dream_tokens)
        if minimum <= 0:
            return keep
        counts = self.group_token_counts.to(keep.device).expand_as(keep)
        needed = (minimum - (keep * counts).sum(dim=-1)).clamp_min(0)
        order = torch.argsort(score, dim=-1, descending=True, stable=True)
        candidates = (~keep).gather(1, order)
        sizes = counts.gather(1, order) * candidates
        add = candidates & ((sizes.cumsum(dim=-1) - sizes) < needed[:, None])
        return keep | torch.zeros_like(keep).scatter(1, order, add)

    def _group_result(self, group_gate, group_keep, group_score):
        ids = self.group_ids.to(group_gate.device)
        return dict(group_gate=group_gate, group_keep=group_keep, group_score=group_score,
                    gate=group_gate[:, ids], keep=group_keep[:, ids], score=group_score[:, ids])

    def gate_for_layer(
        self,
        *,
        layer_idx: int,
        q_action: torch.Tensor,
        k_video_current: Optional[torch.Tensor] = None,
        k_dream: Optional[torch.Tensor] = None,
        dense_probs: Optional[torch.Tensor] = None,
        dream_slice: Optional[slice] = None,
        n_valid: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute one layer's gate; Dream K is a generator input only for DA.

        Returns a dict with:
            gate: [B, Sd] multiplicative gate in [0, 1] (differentiable in
                ``learned`` mode, a hard 0/1 tensor in ``threshold`` mode).
            keep: [B, Sd] boolean mask used for hard pruning / statistics.
            score: [B, Sd] the quantity the decision was based on.
            group_gate/group_keep/group_score: [B, Ng] in learned mode. group_gate
                is the gate actually used, after warmup and optional hard pruning.
        """
        batch_size = int(q_action.shape[0])
        num_dream = self.num_dream_tokens
        if k_dream is not None and k_dream.shape[1] != num_dream:
            raise ValueError("Dream key count does not match the configured token layout.")
        device = q_action.device

        if layer_idx not in self.active_layers or self.config.mode == "none":
            if self.config.mode == "learned":
                group = torch.ones((batch_size, len(self.group_names)), device=device)
                return self._group_result(group, group.bool(), group)
            gate = torch.ones((batch_size, num_dream), device=device, dtype=torch.float32)
            return {"gate": gate, "keep": gate.bool(), "score": gate}

        if self.config.debug_force_gate is not None:
            value = float(self.config.debug_force_gate)
            if self.config.mode == "learned":
                group = torch.full((batch_size, len(self.group_names)), value, device=device)
                return self._group_result(group, group > self.config.gate_threshold, group)
            gate = torch.full((batch_size, num_dream), value, device=device, dtype=torch.float32)
            return {"gate": gate, "keep": gate > self.config.gate_threshold, "score": gate}

        if self.config.mode == "threshold":
            if dense_probs is None or dream_slice is None or n_valid is None:
                raise ValueError(
                    "router.mode='threshold' needs dense_probs, dream_slice and n_valid."
                )
            score = self._uniform_normalised_dream_mass(
                dense_probs=dense_probs, dream_slice=dream_slice, n_valid=n_valid
            ).detach()
            keep = score >= float(self.config.alpha)
            keep = self._apply_min_keep(keep, score)
            gate = keep.to(dtype=torch.float32)
        else:
            group_score = self.learned_group_gate(
                layer_idx=layer_idx, q_action=q_action, k_video_current=k_video_current, k_dream=k_dream
            )
            strength = self.current_strength()
            if strength < 1.0:
                # Blend towards the dense gate of 1 during warmup so the model
                # does not have to survive a discontinuity at step 0.
                group_score = group_score * strength + (1.0 - strength)
            group_keep = group_score > float(self.config.gate_threshold)
            group_keep = self._apply_group_min_keep(group_keep, group_score.detach())
            group_gate = group_score
            if self.config.hard_prune_at_inference and not self.training:
                group_gate = group_gate * group_keep.to(dtype=group_gate.dtype)
            return self._group_result(group_gate, group_keep, group_score)

        return {"gate": gate, "keep": keep, "score": score}

    def record(
        self,
        *,
        layer_idx: int,
        gate: torch.Tensor,
        keep: torch.Tensor,
        score: torch.Tensor,
        group_gate: Optional[torch.Tensor] = None,
        group_keep: Optional[torch.Tensor] = None,
        group_score: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.config.log_statistics:
            return
        with torch.no_grad():
            group_ids = self.group_ids.to(device=keep.device)
            per_group = {}
            per_group_gate = {}
            for index, name in enumerate(self.group_names):
                selector = group_ids == index
                if not bool(selector.any()):
                    continue
                per_group[name] = float((group_keep[:, index] if group_keep is not None
                                         else keep[:, selector]).float().mean().item())
                per_group_gate[name] = float((group_gate[:, index] if group_gate is not None
                                              else gate[:, selector]).float().mean().item())
            self.last_statistics.append(
                {
                    "layer": int(layer_idx),
                    "keep_ratio": float((group_keep if group_keep is not None else keep).float().mean().item()),
                    "gate_mean": float((group_gate if group_gate is not None else gate).float().mean().item()),
                    "score_mean": float((group_score if group_score is not None else score).float().mean().item()),
                    "token_keep_ratio": float(keep.float().mean().item()),
                    "k_mean": float(keep.sum(dim=-1).float().mean().item()),
                    "per_group_keep_ratio": per_group,
                    "per_group_gate_mean": per_group_gate,
                }
            )

    def scalar_metrics(self) -> dict[str, float]:
        if not self.last_statistics:
            return {}
        count = float(len(self.last_statistics))
        metrics = {
            "router_keep_ratio": sum(r["keep_ratio"] for r in self.last_statistics) / count,
            "router_gate_mean": sum(r["gate_mean"] for r in self.last_statistics) / count,
            "router_k_mean": sum(r["k_mean"] for r in self.last_statistics) / count,
        }
        for name in self.group_names:
            values = [
                r["per_group_keep_ratio"][name]
                for r in self.last_statistics
                if name in r["per_group_keep_ratio"]
            ]
            if values:
                metrics[f"router_keep_{name}"] = sum(values) / len(values)
                metrics[f"router_gate_{name}"] = sum(
                    r["per_group_gate_mean"][name] for r in self.last_statistics
                ) / count
        return metrics


class DynamicFeatureRouter(nn.Module):
    """One semantic gate per modality/view/horizon, shared by the entire action solve.

    Legacy ImaginationRouter remains available for old experiments. This module
    never reads action noise, prunes tokens, or normalizes across groups.
    """

    modality_order = ("dino", "dyn", "sam", "depth")
    activation_names = ("dino_activation", "tracker_activation", "sam_activation", "depth_activation")

    def __init__(self, *, config: RouterConfig, input_dim: int, dream_expert):
        super().__init__()
        self.config = config
        if config.routing_mode is None:
            raise ValueError("DynamicFeatureRouter requires an explicit routing_mode.")
        if set(dream_expert.modalities) != set(self.modality_order):
            raise ValueError("16-group routing requires dyn, depth, dino, sam.")
        split = dream_expert.camera_token_split
        if split is None or len(split) != 2 or len(set(dream_expert.future_offsets)) != 2:
            raise ValueError("16-group routing requires two camera groups and two future offsets.")
        self.group_mapping = []
        token_groups = []
        for modality in dream_expert.modalities:
            if sum(split) != getattr(dream_expert, f"n_{modality}"):
                raise ValueError("Camera token split does not match the Dream layout.")
            for offset in dream_expert.future_offsets:
                for view, count in zip(("primary", "wrist"), split):
                    group_id = len(self.group_mapping)
                    start = len(token_groups)
                    token_groups.extend([group_id] * count)
                    self.group_mapping.append(dict(
                        group_id=group_id, modality=modality, view=view,
                        future_offset=int(offset), token_start=start, token_stop=len(token_groups),
                    ))
        if len(self.group_mapping) != 16:
            raise ValueError("Expected exactly 16 semantic groups.")
        self.register_buffer("group_ids", torch.tensor(token_groups, dtype=torch.long), persistent=False)
        self.register_buffer("modality_indices", torch.tensor([
            [g["group_id"] for g in self.group_mapping if g["modality"] == m]
            for m in self.modality_order
        ]), persistent=False)
        self.gate_prior = nn.Parameter(torch.zeros(16))
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, config.router_hidden_dim),
            nn.GELU(), nn.Linear(config.router_hidden_dim, 16),
        ) if config.routing_mode == "dynamic" else None
        if self.network is not None:
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        self.global_step = 0
        if self.full:
            self.requires_grad_(False)

    def _apply(self, fn, recurse=True):
        # DeepSpeed calls model.bfloat16() before building ZeRO partitions.
        # Keep the small semantic router in FP32, including through parent .to().
        # Use the original tensor when converting back to avoid BF16 rounding.
        def preserve_precision(tensor):
            converted = fn(tensor)
            if tensor.is_floating_point():
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted
        return super()._apply(preserve_precision, recurse=recurse)

    @torch.no_grad()
    def reset_zero(self):
        """Fresh stage-two router: every input starts with zero Dream QK scale."""
        self.gate_prior.zero_()
        if self.network is not None:
            for module in self.network:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    @property
    def full(self):
        return not self.config.router_enabled or self.config.routing_mode == "full"

    def warming_up(self, training: bool) -> bool:
        return training and self.global_step < self.config.router_warmup_steps

    def forward(self, context: torch.Tensor, *, training: bool = False) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError("Router context must be [B,D].")
        if self.full:
            return context.new_ones((context.shape[0], 16), dtype=torch.float32)
        # A zero dependency keeps DDP's parameter participation stable across
        # warmup, without allowing either action or gate loss to train gates yet.
        logits = self.gate_prior.expand(context.shape[0], -1)
        if self.network is not None:
            with torch.autocast(device_type=context.device.type, enabled=False):
                logits = logits + self.network(context.float())
        if self.warming_up(training):
            return logits.float() * 0.0
        return torch.tanh(logits.float())

    def gate_loss(self, gates: torch.Tensor, *, training: bool) -> torch.Tensor:
        if self.full or self.warming_up(training):
            return gates.sum() * 0.0
        return gates.abs().mean()

    def activations(self, gates: torch.Tensor) -> torch.Tensor:
        if gates.ndim != 2 or gates.shape[1] != 16:
            raise ValueError("Group gates must be [B,16].")
        return gates[:, self.modality_indices].mean(dim=-1)

    def gate_cache(self, cache: dict, gates: torch.Tensor) -> dict:
        """Attach QK scales for Action; preserve all raw K/V tensors."""
        activations = self.activations(gates)  # also validate the public gate shape
        nv, nd = cache["video_seq_len"], cache["dream_seq_len"]
        if nd != self.group_ids.numel():
            raise ValueError("Dream cache token count does not match the group mapping.")
        weights = gates[:, self.group_ids]
        bypass = self.full and bool(torch.all(gates == 1))
        layers = []
        for layer in cache["kv_cache"]:
            value = layer["v"]
            if value.shape[1] != nv + nd or value.shape[0] != gates.shape[0]:
                raise ValueError("Action context cache shape does not match gates/layout.")
            layers.append(dict(layer) if bypass else {**layer, "dream_token_gates": weights})
        return {**cache, "kv_cache": layers, "group_gates": gates,
                "modality_activations": activations,
                "group_mapping": self.group_mapping}
