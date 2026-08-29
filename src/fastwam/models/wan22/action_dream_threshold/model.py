from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from ..dream_fastwam.model import DreamFastWAM
from ..fastwam.model import FastWAM
from ..mot import MoT
from ..wan_video_dit import flash_attention


logger = get_logger(__name__)


def rope_pair_uniform_channel_indices(
    full_head_dim: int,
    proxy_head_dim: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Select uniformly spaced complete RoPE pairs from one attention head."""
    full_head_dim = int(full_head_dim)
    proxy_head_dim = int(proxy_head_dim)
    if full_head_dim <= 0 or full_head_dim % 2 != 0:
        raise ValueError(f"full_head_dim must be a positive even integer, got {full_head_dim}.")
    if proxy_head_dim <= 0 or proxy_head_dim > full_head_dim:
        raise ValueError(
            f"proxy_head_dim must be in (0, {full_head_dim}], got {proxy_head_dim}."
        )
    if proxy_head_dim % 2 != 0:
        raise ValueError(f"proxy_head_dim must be even, got {proxy_head_dim}.")

    full_pairs = full_head_dim // 2
    proxy_pairs = proxy_head_dim // 2
    pair_indices = torch.div(
        torch.arange(proxy_pairs, dtype=torch.long, device=device) * full_pairs,
        proxy_pairs,
        rounding_mode="floor",
    )
    return torch.stack((pair_indices * 2, pair_indices * 2 + 1), dim=-1).flatten()


@dataclass(frozen=True)
class ActionDreamThresholdConfig:
    """Non-parametric policy for Action-to-Dream mixed-attention pruning."""

    enabled: bool = False
    alpha: float = 1.0
    warmup_ratio: float = 0.1
    head_reduce: str = "mean"
    action_query_reduce: str = "max"
    min_keep_dream_tokens: int = 0
    detach_selection_score: bool = True
    log_statistics: bool = True
    save_detailed_tensors: bool = False
    proxy_analysis_enabled: bool = False
    proxy_head_dim: int = 16
    proxy_channel_sampling: str = "rope_pair_uniform"

    @classmethod
    def from_dict(cls, value: Optional[dict[str, Any]]) -> "ActionDreamThresholdConfig":
        cfg = cls(**({} if value is None else dict(value)))
        if cfg.alpha < 0:
            raise ValueError(f"action_dream_threshold.alpha must be >= 0, got {cfg.alpha}.")
        if not 0.0 <= cfg.warmup_ratio <= 1.0:
            raise ValueError(
                "action_dream_threshold.warmup_ratio must be in [0,1], "
                f"got {cfg.warmup_ratio}."
            )
        if cfg.head_reduce != "mean":
            raise ValueError("The first implementation supports head_reduce='mean' only.")
        if cfg.action_query_reduce != "max":
            raise ValueError("The first implementation supports action_query_reduce='max' only.")
        if cfg.min_keep_dream_tokens < 0:
            raise ValueError("min_keep_dream_tokens must be >= 0.")
        if not cfg.detach_selection_score:
            raise ValueError("The hard-threshold implementation requires detach_selection_score=true.")
        if cfg.proxy_head_dim <= 0:
            raise ValueError("action_dream_threshold.proxy_head_dim must be positive.")
        if cfg.proxy_head_dim % 2 != 0:
            raise ValueError("action_dream_threshold.proxy_head_dim must be even.")
        if cfg.proxy_channel_sampling != "rope_pair_uniform":
            raise ValueError(
                "The proxy implementation supports proxy_channel_sampling="
                "'rope_pair_uniform' only."
            )
        return cfg


class ActionDreamThresholdMoT(MoT):
    """MoT variant that thresholds only Action-query to Dream-key edges.

    Video, Dream, and Action keys remain in one shared mixed-attention
    distribution.  Only the Action query rows are materialized explicitly;
    non-Action rows continue to use PyTorch SDPA.
    """

    def __init__(
        self,
        mixtures: Dict[str, torch.nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        action_dream_threshold: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            mixtures=mixtures,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        self.threshold_config = ActionDreamThresholdConfig.from_dict(action_dream_threshold)
        self._alpha_current = float(self.threshold_config.alpha)
        self._threshold_global_step = 0
        self._threshold_total_steps = 0
        self._collect_training_statistics = False
        self.last_threshold_statistics: list[dict[str, Any]] = []
        if self.threshold_config.proxy_analysis_enabled:
            rope_pair_uniform_channel_indices(
                self.attn_head_dim,
                self.threshold_config.proxy_head_dim,
            )

    def set_training_progress(
        self,
        global_step: int,
        total_steps: int,
        collect_statistics: bool = False,
    ) -> None:
        self._threshold_global_step = max(int(global_step), 0)
        self._threshold_total_steps = max(int(total_steps), 0)
        self._collect_training_statistics = bool(collect_statistics)
        cfg = self.threshold_config
        if not self.training or cfg.warmup_ratio <= 0.0 or self._threshold_total_steps <= 0:
            self._alpha_current = float(cfg.alpha)
            return
        warmup_steps = max(int(round(self._threshold_total_steps * cfg.warmup_ratio)), 1)
        fraction = min(float(self._threshold_global_step) / float(warmup_steps), 1.0)
        self._alpha_current = float(cfg.alpha) * fraction

    def current_alpha(self) -> float:
        if not self.training:
            return float(self.threshold_config.alpha)
        return float(self._alpha_current)

    @staticmethod
    def _expert_slices(expert_order: list[str], seq_lens: list[int]) -> dict[str, slice]:
        result: dict[str, slice] = {}
        start = 0
        for name, length in zip(expert_order, seq_lens):
            end = start + int(length)
            result[name] = slice(start, end)
            start = end
        return result

    @staticmethod
    def _action_mask_broadcast(
        attention_mask: torch.Tensor,
        action_slice: slice,
        *,
        batch_size: int,
        num_heads: int,
        total_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a boolean allowed-key mask shaped [B,H,Sa,S]."""
        mask = attention_mask.to(device=device)
        if mask.dtype is not torch.bool:
            raise TypeError("Action-Dream threshold currently requires a boolean attention mask.")
        action_len = int(action_slice.stop - action_slice.start)
        if mask.ndim == 2:
            if tuple(mask.shape) != (total_seq_len, total_seq_len):
                raise ValueError("2D attention mask does not match the mixed sequence length.")
            mask = mask[action_slice, :].view(1, 1, action_len, total_seq_len)
        elif mask.ndim == 3:
            if mask.shape[-2:] != (total_seq_len, total_seq_len):
                raise ValueError("3D attention mask does not match the mixed sequence length.")
            if mask.shape[0] not in (1, batch_size):
                raise ValueError("3D attention mask batch dimension is not broadcastable.")
            mask = mask[:, action_slice, :].unsqueeze(1)
        elif mask.ndim == 4:
            if mask.shape[-2:] != (total_seq_len, total_seq_len):
                raise ValueError("4D attention mask does not match the mixed sequence length.")
            if mask.shape[0] not in (1, batch_size) or mask.shape[1] not in (1, num_heads):
                raise ValueError("4D attention mask batch/head dimensions are not broadcastable.")
            mask = mask[:, :, action_slice, :]
        else:
            raise ValueError(f"Unsupported attention mask rank {mask.ndim}; expected 2, 3, or 4.")
        return mask.expand(batch_size, num_heads, action_len, total_seq_len)

    def _apply_minimum_keep(self, keep: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        minimum = min(int(self.threshold_config.min_keep_dream_tokens), int(score.shape[-1]))
        if minimum <= 0:
            return keep
        needed = (minimum - keep.sum(dim=-1)).clamp(min=0)
        if not bool((needed > 0).any()):
            return keep
        # This is an explicitly requested minimum-retention fallback, not the
        # default selection policy.  The default minimum is zero.
        order = torch.argsort(score, dim=-1, descending=True, stable=True)
        ranks = torch.arange(score.shape[-1], device=score.device).view(1, -1)
        additions_in_rank_order = ranks < needed.unsqueeze(-1)
        additions = torch.zeros_like(keep).scatter(1, order, additions_in_rank_order)
        return keep | additions

    def _threshold_action_attention(
        self,
        q_action: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        attention_mask: torch.Tensor,
        action_slice: slice,
        dream_slice: slice,
        alpha: float,
    ) -> tuple[torch.Tensor, ...]:
        bsz, action_len, inner_dim = q_action.shape
        total_seq_len = int(k_all.shape[1])
        expected_inner = self.num_heads * self.attn_head_dim
        if inner_dim != expected_inner or k_all.shape[-1] != expected_inner:
            raise ValueError(
                f"Threshold attention expected inner dim {expected_inner}, "
                f"got q={inner_dim}, k={k_all.shape[-1]}."
            )

        q = q_action.reshape(bsz, action_len, self.num_heads, self.attn_head_dim).transpose(1, 2)
        k = k_all.reshape(bsz, total_seq_len, self.num_heads, self.attn_head_dim).transpose(1, 2)
        v = v_all.reshape(bsz, total_seq_len, self.num_heads, self.attn_head_dim).transpose(1, 2)
        allowed = self._action_mask_broadcast(
            attention_mask,
            action_slice,
            batch_size=bsz,
            num_heads=self.num_heads,
            total_seq_len=total_seq_len,
            device=q.device,
        )

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        scores.mul_(self.attn_head_dim ** -0.5)
        dense_scores = scores.masked_fill(~allowed, -torch.inf)
        dense_probs = torch.softmax(dense_scores, dim=-1)
        if not bool(torch.isfinite(dense_probs).all()):
            raise FloatingPointError("Dense Action mixed-attention probabilities contain NaN/Inf.")

        valid_counts_by_head = allowed.sum(dim=-1)
        if not torch.equal(
            valid_counts_by_head,
            valid_counts_by_head[:, :1].expand_as(valid_counts_by_head),
        ):
            raise ValueError("N_valid differs by attention head; score definition requires [B,Action].")
        n_valid = valid_counts_by_head[:, 0].to(dtype=dense_probs.dtype)

        dense_dream = dense_probs[..., dream_slice]
        normalized = dense_dream * n_valid[:, None, :, None]
        score = normalized.mean(dim=1).amax(dim=1)
        selection_score = score.detach()
        keep = selection_score >= float(alpha)
        keep = self._apply_minimum_keep(keep, selection_score)

        pruned_allowed = allowed.clone()
        pruned_allowed[..., dream_slice] &= keep[:, None, None, :]
        pruned_scores = scores.masked_fill(~pruned_allowed, -torch.inf)
        pruned_probs = torch.softmax(pruned_scores, dim=-1)
        if not bool(torch.isfinite(pruned_probs).all()):
            raise FloatingPointError("Pruned Action mixed-attention probabilities contain NaN/Inf.")

        action_out = torch.matmul(pruned_probs.to(dtype=v.dtype), v)
        action_out = action_out.transpose(1, 2).reshape(bsz, action_len, inner_dim)

        dense_mass = dense_dream.sum(dim=-1).mean(dim=(1, 2))
        kept_mass = (dense_dream * keep[:, None, None, :]).sum(dim=-1).mean(dim=(1, 2))
        pruned_mass = pruned_probs[..., dream_slice].sum(dim=-1).mean(dim=(1, 2))
        proxy_analysis = None
        if self.threshold_config.proxy_analysis_enabled:
            # Analysis is deliberately evaluated only after the real Action
            # output has been computed from the full-dimensional teacher mask.
            # No proxy tensor participates in ``pruned_allowed`` or action_out.
            with torch.no_grad():
                proxy_head_dim = int(self.threshold_config.proxy_head_dim)
                channels = rope_pair_uniform_channel_indices(
                    self.attn_head_dim,
                    proxy_head_dim,
                    device=q.device,
                )
                q_proxy = q.index_select(-1, channels)
                k_proxy = k.index_select(-1, channels)
                proxy_logits = torch.matmul(
                    q_proxy.float(),
                    k_proxy.float().transpose(-2, -1),
                )
                proxy_logits.mul_(proxy_head_dim ** -0.5)
                proxy_probs = torch.softmax(
                    proxy_logits.masked_fill(~allowed, -torch.inf),
                    dim=-1,
                )
                if not bool(torch.isfinite(proxy_probs).all()):
                    raise FloatingPointError(
                        "Proxy Action mixed-attention probabilities contain NaN/Inf."
                    )
                proxy_dream = proxy_probs[..., dream_slice]
                proxy_normalized = proxy_dream * n_valid[:, None, :, None]
                proxy_score = proxy_normalized.mean(dim=1).amax(dim=1).detach()
                proxy_keep = proxy_score >= float(alpha)
                proxy_keep = self._apply_minimum_keep(proxy_keep, proxy_score)

                full_k = keep.sum(dim=-1)
                proxy_k = proxy_keep.sum(dim=-1)
                intersection = keep & proxy_keep
                union = keep | proxy_keep
                intersection_k = intersection.sum(dim=-1)
                union_k = union.sum(dim=-1)
                full_k_float = full_k.float()
                proxy_k_float = proxy_k.float()
                intersection_float = intersection_k.float()
                union_float = union_k.float()
                ones = torch.ones_like(intersection_float)
                recall = torch.where(
                    full_k > 0,
                    intersection_float / full_k_float.clamp_min(1.0),
                    ones,
                )
                precision = torch.where(
                    proxy_k > 0,
                    intersection_float / proxy_k_float.clamp_min(1.0),
                    ones,
                )
                jaccard = torch.where(
                    union_k > 0,
                    intersection_float / union_float.clamp_min(1.0),
                    ones,
                )
                score_error = proxy_score.float() - selection_score.float()
                proxy_kept_full_mass = (
                    dense_dream * proxy_keep[:, None, None, :]
                ).sum(dim=-1).mean(dim=(1, 2))
                mass_ones = torch.ones_like(dense_mass.float())
                proxy_full_mass_retention = torch.where(
                    dense_mass.float() > 0,
                    proxy_kept_full_mass.float() / dense_mass.float().clamp_min(1.0e-12),
                    mass_ones,
                )
                proxy_analysis = {
                    "proxy_head_dim": proxy_head_dim,
                    "proxy_channel_indices": channels.detach(),
                    "full_score": selection_score.detach(),
                    "proxy_score": proxy_score,
                    "full_keep_mask": keep.detach(),
                    "proxy_keep_mask": proxy_keep.detach(),
                    "full_k": full_k.detach(),
                    "proxy_k": proxy_k.detach(),
                    "k_delta": (proxy_k - full_k).detach(),
                    "teacher_recall": recall.detach(),
                    "teacher_precision": precision.detach(),
                    "teacher_jaccard": jaccard.detach(),
                    "mask_agreement": (keep == proxy_keep).float().mean(dim=-1).detach(),
                    "false_negative_count": (keep & ~proxy_keep).sum(dim=-1).detach(),
                    "false_positive_count": (proxy_keep & ~keep).sum(dim=-1).detach(),
                    "score_mae": score_error.abs().mean(dim=-1).detach(),
                    "score_rmse": score_error.square().mean(dim=-1).sqrt().detach(),
                    "full_dense_dream_mass": dense_mass.detach(),
                    "full_teacher_kept_mass": kept_mass.detach(),
                    "proxy_mask_kept_full_mass": proxy_kept_full_mass.detach(),
                    "proxy_full_mass_retention": proxy_full_mass_retention.detach(),
                }
        return (
            action_out,
            selection_score,
            keep,
            n_valid.detach(),
            dense_mass.detach(),
            kept_mass.detach(),
            pruned_mass.detach(),
            proxy_analysis,
        )

    @staticmethod
    def _record_statistics(
        *,
        layer_idx: int,
        alpha: float,
        slices: dict[str, slice],
        score: torch.Tensor,
        keep: torch.Tensor,
        n_valid: torch.Tensor,
        dense_mass: torch.Tensor,
        kept_mass: torch.Tensor,
        pruned_mass: torch.Tensor,
        proxy_analysis: Optional[dict[str, Any]],
        detailed: bool,
    ) -> dict[str, Any]:
        score_float = score.float()
        k = keep.sum(dim=-1)
        quantiles = torch.quantile(
            score_float.flatten(),
            torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=score.device),
        )
        record: dict[str, Any] = {
            "layer": int(layer_idx),
            "alpha_current": float(alpha),
            "k_mean": float(k.float().mean().item()),
            "k_std": float(k.float().std(unbiased=False).item()),
            "k_min": int(k.min().item()),
            "k_max": int(k.max().item()),
            "dream_keep_ratio": float(keep.float().mean().item()),
            "dense_dream_attention_mass": float(dense_mass.float().mean().item()),
            "kept_dense_dream_attention_mass": float(kept_mass.float().mean().item()),
            "removed_dense_dream_attention_mass": float((dense_mass - kept_mass).float().mean().item()),
            "pruned_dream_attention_mass": float(pruned_mass.float().mean().item()),
            "score_mean": float(score_float.mean().item()),
            "score_std": float(score_float.std(unbiased=False).item()),
            "score_q10": float(quantiles[0].item()),
            "score_q25": float(quantiles[1].item()),
            "score_q50": float(quantiles[2].item()),
            "score_q75": float(quantiles[3].item()),
            "score_q90": float(quantiles[4].item()),
            "n_valid_mean": float(n_valid.float().mean().item()),
            "slices": {name: [slc.start, slc.stop] for name, slc in slices.items()},
        }
        if detailed:
            record.update(
                {
                    "k_per_sample": k.detach().cpu(),
                    "score": score.detach().cpu(),
                    "keep_mask": keep.detach().cpu(),
                    "dense_dream_mass_per_sample": dense_mass.detach().cpu(),
                    "kept_dense_dream_mass_per_sample": kept_mass.detach().cpu(),
                    "pruned_dream_mass_per_sample": pruned_mass.detach().cpu(),
                }
            )
        if proxy_analysis is not None:
            record.update(
                {
                    "proxy_head_dim": int(proxy_analysis["proxy_head_dim"]),
                    "full_k_mean": float(proxy_analysis["full_k"].float().mean().item()),
                    "proxy_k_mean": float(proxy_analysis["proxy_k"].float().mean().item()),
                    "proxy_k_delta_mean": float(
                        proxy_analysis["k_delta"].float().mean().item()
                    ),
                    "proxy_teacher_recall": float(
                        proxy_analysis["teacher_recall"].float().mean().item()
                    ),
                    "proxy_teacher_precision": float(
                        proxy_analysis["teacher_precision"].float().mean().item()
                    ),
                    "proxy_teacher_jaccard": float(
                        proxy_analysis["teacher_jaccard"].float().mean().item()
                    ),
                    "proxy_mask_agreement": float(
                        proxy_analysis["mask_agreement"].float().mean().item()
                    ),
                    "proxy_false_negative_mean": float(
                        proxy_analysis["false_negative_count"].float().mean().item()
                    ),
                    "proxy_false_positive_mean": float(
                        proxy_analysis["false_positive_count"].float().mean().item()
                    ),
                    "proxy_score_mae": float(
                        proxy_analysis["score_mae"].float().mean().item()
                    ),
                    "proxy_score_rmse": float(
                        proxy_analysis["score_rmse"].float().mean().item()
                    ),
                    "full_dense_dream_mass": float(
                        proxy_analysis["full_dense_dream_mass"].float().mean().item()
                    ),
                    "full_teacher_kept_mass": float(
                        proxy_analysis["full_teacher_kept_mass"].float().mean().item()
                    ),
                    "proxy_mask_kept_full_mass": float(
                        proxy_analysis["proxy_mask_kept_full_mass"].float().mean().item()
                    ),
                    "proxy_full_mass_retention": float(
                        proxy_analysis["proxy_full_mass_retention"].float().mean().item()
                    ),
                }
            )
            if detailed:
                record.update(
                    {
                        "proxy_channel_indices": proxy_analysis[
                            "proxy_channel_indices"
                        ].cpu(),
                        "full_score": record["score"],
                        "proxy_score": proxy_analysis["proxy_score"].cpu(),
                        "full_keep_mask": record["keep_mask"],
                        "proxy_keep_mask": proxy_analysis["proxy_keep_mask"].cpu(),
                        "full_k_per_sample": proxy_analysis["full_k"].cpu(),
                        "proxy_k_per_sample": proxy_analysis["proxy_k"].cpu(),
                        "proxy_k_delta_per_sample": proxy_analysis["k_delta"].cpu(),
                        "proxy_teacher_recall_per_sample": proxy_analysis[
                            "teacher_recall"
                        ].cpu(),
                        "proxy_teacher_precision_per_sample": proxy_analysis[
                            "teacher_precision"
                        ].cpu(),
                        "proxy_teacher_jaccard_per_sample": proxy_analysis[
                            "teacher_jaccard"
                        ].cpu(),
                        "proxy_mask_agreement_per_sample": proxy_analysis[
                            "mask_agreement"
                        ].cpu(),
                        "proxy_false_negative_per_sample": proxy_analysis[
                            "false_negative_count"
                        ].cpu(),
                        "proxy_false_positive_per_sample": proxy_analysis[
                            "false_positive_count"
                        ].cpu(),
                        "proxy_score_mae_per_sample": proxy_analysis["score_mae"].cpu(),
                        "proxy_score_rmse_per_sample": proxy_analysis[
                            "score_rmse"
                        ].cpu(),
                        "full_dense_dream_mass_per_sample": record[
                            "dense_dream_mass_per_sample"
                        ],
                        "full_teacher_kept_mass_per_sample": record[
                            "kept_dense_dream_mass_per_sample"
                        ],
                        "proxy_mask_kept_full_mass_per_sample": proxy_analysis[
                            "proxy_mask_kept_full_mass"
                        ].cpu(),
                        "proxy_full_mass_retention_per_sample": proxy_analysis[
                            "proxy_full_mass_retention"
                        ].cpu(),
                    }
                )
        return record

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        return_action_attention: bool = False,
        attention_layers: Optional[list[int]] = None,
        return_threshold_statistics: bool = False,
    ):
        cfg = self.threshold_config
        if not cfg.enabled:
            self.last_threshold_statistics = []
            return super().forward(
                embeds_all=embeds_all,
                attention_mask=attention_mask,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
                return_action_attention=return_action_attention,
                attention_layers=attention_layers,
            )

        missing = [name for name in self.expert_order if name not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}.")
        if "dream" not in self.expert_order or "action" not in self.expert_order:
            raise ValueError("Action-Dream threshold requires both dream and action experts.")
        if attention_mask.ndim != 2:
            raise ValueError("The current DreamFastWAM forward expects a 2D mixed attention mask.")

        alpha = self.current_alpha()
        # Exact dense fast path, including the first warmup step.
        if alpha <= 0.0 and cfg.min_keep_dream_tokens == 0:
            self.last_threshold_statistics = []
            return super().forward(
                embeds_all=embeds_all,
                attention_mask=attention_mask,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
                return_action_attention=return_action_attention,
                attention_layers=attention_layers,
            )

        tokens_all = dict(embeds_all)
        attention_layers_set = None if attention_layers is None else {int(x) for x in attention_layers}
        action_attention_records: list[dict[str, Any]] = []
        threshold_records: list[dict[str, Any]] = []
        collect_statistics = bool(
            return_action_attention
            or return_threshold_statistics
            or cfg.save_detailed_tensors
            or cfg.proxy_analysis_enabled
            or (cfg.log_statistics and self._collect_training_statistics)
        )

        for layer_idx in range(self.num_layers):
            q_chunks: list[torch.Tensor] = []
            k_chunks: list[torch.Tensor] = []
            v_chunks: list[torch.Tensor] = []
            cached: dict[str, dict[str, Any]] = {}
            seq_lens: list[int] = []
            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                io = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs_all[name],
                    t_mod=t_mod_all[name],
                )
                q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gc = io
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(int(x.shape[1]))
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gc,
                }

            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)
            slices = self._expert_slices(self.expert_order, seq_lens)
            action_slice = slices["action"]
            dream_slice = slices["dream"]
            total_seq = int(q_cat.shape[1])
            if attention_mask.shape != (total_seq, total_seq):
                raise ValueError("Attention mask sequence length does not match dynamic expert lengths.")
            if action_slice.stop != total_seq:
                raise ValueError(
                    "Threshold execution requires the dynamically located action expert to be the final query chunk."
                )

            # Non-Action query rows retain one mixed key space and use SDPA.
            non_action_end = int(action_slice.start)
            non_action_mask = attention_mask[:non_action_end, :]
            non_action_out = flash_attention(
                q=q_cat[:, :non_action_end],
                k=k_cat,
                v=v_cat,
                num_heads=self.num_heads,
                ctx_mask=non_action_mask,
            )

            def action_fn(q_action, k_all, v_all):
                return self._threshold_action_attention(
                    q_action=q_action,
                    k_all=k_all,
                    v_all=v_all,
                    attention_mask=attention_mask,
                    action_slice=action_slice,
                    dream_slice=dream_slice,
                    alpha=alpha,
                )

            if self.mot_checkpoint_mixed_attn and self.training:
                action_result = torch.utils.checkpoint.checkpoint(
                    action_fn,
                    q_cat[:, action_slice],
                    k_cat,
                    v_cat,
                    use_reentrant=False,
                )
            else:
                action_result = action_fn(q_cat[:, action_slice], k_cat, v_cat)
            (
                action_out,
                score,
                keep,
                n_valid,
                dense_mass,
                kept_mass,
                pruned_mass,
                proxy_analysis,
            ) = action_result
            mixed = torch.cat([non_action_out, action_out], dim=1)

            if collect_statistics:
                detailed = bool(
                    cfg.save_detailed_tensors
                    or return_threshold_statistics
                    or return_action_attention
                )
                threshold_records.append(self._record_statistics(
                    layer_idx=layer_idx,
                    alpha=alpha,
                    slices=slices,
                    score=score,
                    keep=keep,
                    n_valid=n_valid,
                    dense_mass=dense_mass,
                    kept_mass=kept_mass,
                    pruned_mass=pruned_mass,
                    proxy_analysis=proxy_analysis,
                    detailed=detailed,
                ))

            if return_action_attention and (
                attention_layers_set is None or layer_idx in attention_layers_set
            ):
                action_attention_records.append(
                    {
                        "layer": int(layer_idx),
                        "probs": self._compute_action_attention_probs(
                            q_cat=q_cat,
                            k_cat=k_cat,
                            attention_mask=attention_mask,
                            action_slice=action_slice,
                        ),
                        "slices": {
                            name: [slc.start, slc.stop] for name, slc in slices.items()
                        },
                        "threshold": threshold_records[-1],
                    }
                )

            for name in self.expert_order:
                slc = slices[name]
                item = cached[name]
                tokens_all[name] = self._apply_post_with_optional_checkpoint(
                    block=item["block"],
                    residual_x=item["residual_x"],
                    gate_msa=item["gate_msa"],
                    shift_mlp=item["shift_mlp"],
                    scale_mlp=item["scale_mlp"],
                    gate_mlp=item["gate_mlp"],
                    use_gradient_checkpointing=item["use_gradient_checkpointing"],
                    mixed_slice=mixed[:, slc],
                    context_payload=context_all.get(name),
                )

        self.last_threshold_statistics = threshold_records
        if return_action_attention or return_threshold_statistics:
            result: dict[str, Any] = {"tokens": tokens_all}
            if return_action_attention:
                result["action_attention"] = action_attention_records
            if return_threshold_statistics:
                result["action_dream_threshold"] = threshold_records
            return result
        return tokens_all

    def forward_action_with_context_cache(self, *args, **kwargs) -> torch.Tensor:
        self._cached_threshold_records: list[dict[str, Any]] = []
        output = super().forward_action_with_context_cache(*args, **kwargs)
        self.last_threshold_statistics = self._cached_threshold_records
        del self._cached_threshold_records
        return output

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
        alpha = self.current_alpha()
        if not cfg.enabled or (alpha <= 0.0 and cfg.min_keep_dream_tokens == 0):
            return super()._action_attention_with_context_cache(
                q_action=q_action,
                k_all=k_all,
                v_all=v_all,
                attention_mask=attention_mask,
                action_slice=action_slice,
                context_slices=context_slices,
                layer_idx=layer_idx,
            )

        action_result = self._threshold_action_attention(
            q_action=q_action,
            k_all=k_all,
            v_all=v_all,
            attention_mask=attention_mask,
            action_slice=action_slice,
            dream_slice=context_slices["dream"],
            alpha=alpha,
        )
        (
            action_out,
            score,
            keep,
            n_valid,
            dense_mass,
            kept_mass,
            pruned_mass,
            proxy_analysis,
        ) = action_result
        collect_statistics = bool(
            cfg.save_detailed_tensors
            or cfg.proxy_analysis_enabled
            or (cfg.log_statistics and self._collect_training_statistics)
        )
        if collect_statistics:
            slices = dict(context_slices)
            slices["action"] = action_slice
            self._cached_threshold_records.append(self._record_statistics(
                layer_idx=layer_idx,
                alpha=alpha,
                slices=slices,
                score=score,
                keep=keep,
                n_valid=n_valid,
                dense_mass=dense_mass,
                kept_mass=kept_mass,
                pruned_mass=pruned_mass,
                proxy_analysis=proxy_analysis,
                detailed=bool(cfg.save_detailed_tensors),
            ))
        return action_out


class ThresholdDreamFastWAM(DreamFastWAM):
    """DreamFastWAM adapter using :class:`ActionDreamThresholdMoT`.

    The adapter deliberately adds no parameters, so dense DreamFastWAM weight
    checkpoints remain directly loadable.
    """

    @classmethod
    def from_wan22_pretrained(
        cls,
        *args,
        action_dream_threshold: Optional[dict[str, Any]] = None,
        finetune_action_only: bool = False,
        **kwargs,
    ) -> "ThresholdDreamFastWAM":
        model = DreamFastWAM.from_wan22_pretrained(*args, **kwargs)
        return cls.from_dense_model(
            model,
            action_dream_threshold=action_dream_threshold,
            finetune_action_only=finetune_action_only,
        )

    @classmethod
    def from_dense_model(
        cls,
        model: DreamFastWAM,
        *,
        action_dream_threshold: Optional[dict[str, Any]] = None,
        finetune_action_only: bool = False,
    ) -> "ThresholdDreamFastWAM":
        if not isinstance(model, DreamFastWAM):
            raise TypeError(f"Expected DreamFastWAM, got {type(model)}.")
        dense_mot = model.mot
        model.__class__ = cls
        model.finetune_action_only = bool(finetune_action_only)
        model.mot = ActionDreamThresholdMoT(
            mixtures={name: dense_mot.mixtures[name] for name in dense_mot.expert_order},
            mot_checkpoint_mixed_attn=dense_mot.mot_checkpoint_mixed_attn,
            action_dream_threshold=action_dream_threshold,
        )
        model.dit = model.mot
        model._training_progress_provider = None
        logger.info(
            "Installed parameter-free Action-Dream threshold MoT: enabled=%s alpha=%.4f action_only=%s",
            model.mot.threshold_config.enabled,
            model.mot.threshold_config.alpha,
            model.finetune_action_only,
        )
        return model

    def set_training_progress_provider(self, provider) -> None:
        self._training_progress_provider = provider

    def _refresh_threshold_progress(self) -> None:
        if self._training_progress_provider is None:
            return
        progress = tuple(self._training_progress_provider())
        if len(progress) == 2:
            global_step, total_steps = progress
            collect_statistics = False
        elif len(progress) == 3:
            global_step, total_steps, collect_statistics = progress
        else:
            raise ValueError("Training progress provider must return 2 or 3 values.")
        self.mot.set_training_progress(
            global_step=global_step,
            total_steps=total_steps,
            collect_statistics=collect_statistics,
        )

    def configure_trainable_parameters(self, freeze_video_expert: bool = False):
        if not self.finetune_action_only:
            return super().configure_trainable_parameters(freeze_video_expert=freeze_video_expert)
        self.eval()
        self.requires_grad_(False)
        # Keep the threshold MoT itself in training mode so global-step warmup
        # and mixed-attention checkpointing remain active.  Individual frozen
        # experts are put back into eval mode below.
        self.mot.train()
        self.video_expert.eval()
        self.dream_expert.eval()
        self.action_expert.train()
        self.action_expert.requires_grad_(True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.eval()
            self.proprio_encoder.requires_grad_(False)
        params = [parameter for parameter in self.action_expert.parameters() if parameter.requires_grad]
        logger.info(
            "Threshold Action-only fine-tuning: %.3fM trainable Action Expert parameters; "
            "Video, Dream, decoder, and proprio encoder frozen.",
            sum(parameter.numel() for parameter in params) / 1e6,
        )
        return params

    def training_loss(self, sample, tiled: bool = False):
        self._refresh_threshold_progress()
        if not self.finetune_action_only:
            loss, metrics = super().training_loss(sample, tiled=tiled)
            return loss, self._append_threshold_scalar_metrics(metrics)
        return self._action_only_training_loss(sample, tiled=tiled)

    def _annotate_threshold_timestep(self, timestep_action: torch.Tensor) -> None:
        timestep = timestep_action.detach().float()
        for record in self.mot.last_threshold_statistics:
            record["diffusion_timestep_mean"] = float(timestep.mean().item())
            if self.mot.threshold_config.save_detailed_tensors or "k_per_sample" in record:
                record["diffusion_timestep_per_sample"] = timestep.cpu()

    @torch.no_grad()
    def _predict_joint_noise(self, *args, **kwargs):
        result = super()._predict_joint_noise(*args, **kwargs)
        timestep_action = kwargs.get("timestep_action")
        if timestep_action is None and len(args) >= 4:
            timestep_action = args[3]
        if timestep_action is not None:
            self._annotate_threshold_timestep(timestep_action)
        return result

    @torch.no_grad()
    def _predict_action_noise_with_cache(self, *args, **kwargs):
        result = super()._predict_action_noise_with_cache(*args, **kwargs)
        timestep_action = kwargs.get("timestep_action")
        if timestep_action is None and len(args) >= 2:
            timestep_action = args[1]
        if timestep_action is not None:
            self._annotate_threshold_timestep(timestep_action)
        return result

    def _append_threshold_scalar_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if not self.mot.threshold_config.log_statistics:
            return metrics
        records = self.mot.last_threshold_statistics
        if not records:
            return metrics
        metrics = dict(metrics)
        metrics["threshold_alpha"] = float(records[0]["alpha_current"])
        metrics["threshold_keep_ratio"] = sum(
            float(record["dream_keep_ratio"]) for record in records
        ) / len(records)
        metrics["threshold_k_mean"] = sum(float(record["k_mean"]) for record in records) / len(records)
        metrics["threshold_dense_dream_mass"] = sum(
            float(record["dense_dream_attention_mass"]) for record in records
        ) / len(records)
        metrics["threshold_pruned_dream_mass"] = sum(
            float(record["pruned_dream_attention_mass"]) for record in records
        ) / len(records)
        return metrics

    def _action_only_training_loss(self, sample, tiled: bool = False):
        # Bypass DreamFastWAM.build_inputs: Action-only fine-tuning does not
        # require decoder supervision or future targets.
        inputs = FastWAM.build_inputs(self, sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = int(input_latents.shape[0])
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]

        current_frame_latents = inputs["first_frame_latents"]
        if current_frame_latents is None:
            current_frame_latents = input_latents[:, :, 0:1]
        timestep_video = torch.zeros(
            (batch_size,), device=self.device, dtype=input_latents.dtype
        )
        noise_action = self._sample_action_noise(
            action, use_correlated_noise=self.use_correlated_noise_train
        )
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=current_frame_latents,
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
        mot_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "dream": dream_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=self._build_mot_attention_mask(
                video_seq_len=video_pre["tokens"].shape[1],
                dream_seq_len=dream_pre["tokens"].shape[1],
                action_seq_len=action_pre["tokens"].shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            ),
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
        self._annotate_threshold_timestep(timestep_action)
        pred_action = self.action_expert.post_dit(mot_out["action"], action_pre)
        token_loss = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is None:
            per_sample = token_loss.mean(dim=1)
        else:
            valid = (~action_is_pad).to(device=token_loss.device, dtype=token_loss.dtype)
            per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        weight = self.train_action_scheduler.training_weight(timestep_action).to(
            device=per_sample.device, dtype=per_sample.dtype
        )
        loss_action = (per_sample * weight).mean()
        metrics = {"loss_action": float(loss_action.detach().item())}
        metrics = self._append_threshold_scalar_metrics(metrics)
        if self.mot.threshold_config.log_statistics and self.mot.last_threshold_statistics:
            metrics["threshold_timestep_mean"] = float(timestep_action.detach().float().mean().item())
        return loss_action, metrics
