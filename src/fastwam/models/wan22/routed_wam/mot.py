"""MoT variant that routes the Action -> Dream attention edges.

Two things are added on top of :class:`fastwam.models.wan22.mot.MoT`:

1. **Routing.**  Action query rows are materialised explicitly. Semantic tanh
   gates multiply Dream logits; ImaginationRouter gates add log(gate). Every other query row
   keeps the original fused SDPA path, so the cost of routing is bounded by the
   action chunk length (32 tokens), not by the mixed sequence length.

2. **A Dream-only forward against a cached Video K/V.**  The Video expert is
   frozen and sees a single clean frame, so its per-layer K/V do not depend on
   the diffusion step.  Only the Dream K/V evolve.  Splitting the prefill into
   "Video once" + "Dream per step" is what makes a multi-step generative Dream
   affordable, and it is also what gives interface distillation a teacher and a
   student that differ only in the number of Dream steps.

Nothing in this module mutates the base class; ``RoutedMoT`` is installed by
swapping ``model.mot`` the same way ``ActionDreamThresholdMoT`` is.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from contextlib import nullcontext

import torch

from fastwam.utils.logging_config import get_logger

from ..mot import MoT
from ..wan_video_dit import flash_attention
from .router import ImaginationRouter


logger = get_logger(__name__)

_LOG_EPS = 1.0e-20


class RoutedMoT(MoT):
    """MoT with an action-side imagination router and a split Video/Dream prefill."""

    def __init__(
        self,
        mixtures: Dict[str, torch.nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        router: Optional[ImaginationRouter] = None,
    ):
        super().__init__(
            mixtures=mixtures,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        if "dream" not in self.expert_order:
            raise ValueError("RoutedMoT requires a 'dream' expert.")
        if self.expert_order[-1] != "action":
            raise ValueError(
                f"RoutedMoT requires Action to be the final expert; got {self.expert_order}."
            )
        self.router = router
        self.capture_kv = False
        self.captured_kv: list[dict[str, torch.Tensor]] = []
        self.last_gates: list[torch.Tensor] = []
        # Analysis-only, default off: records where the action queries actually
        # spend their attention, which is the one quantity the router's own
        # statistics cannot report (a kept token may still be ignored).
        self.capture_action_attention = False
        self.captured_action_attention: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ utils
    @property
    def routing_enabled(self) -> bool:
        return self.router is not None and self.router.config.enabled

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
    def _broadcast_action_mask(
        attention_mask: torch.Tensor,
        action_slice: slice,
        *,
        batch_size: int,
        num_heads: int,
        total_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Boolean allowed-key mask for the action rows, shaped [B,H,Sa,S]."""
        mask = attention_mask.to(device=device)
        if mask.dtype is not torch.bool:
            raise TypeError("RoutedMoT requires a boolean attention mask.")
        if mask.ndim != 2 or tuple(mask.shape) != (total_seq_len, total_seq_len):
            raise ValueError(
                f"Expected a [{total_seq_len},{total_seq_len}] mask, got {tuple(mask.shape)}."
            )
        action_len = int(action_slice.stop - action_slice.start)
        mask = mask[action_slice, :].view(1, 1, action_len, total_seq_len)
        return mask.expand(batch_size, num_heads, action_len, total_seq_len)

    # -------------------------------------------------------------- attention
    @staticmethod
    def _current_video_keys(k_all, attention_mask, action_slice, video_slice):
        """Select the current observation using the same mask Action actually reads.

        DreamFastWAM builds these columns from min(tokens_per_frame, video_seq_len).
        Check the Video rows too: slicing current keys alone would not prevent
        indirect future leakage through bidirectional Video self-attention.
        """
        mask = attention_mask.to(device=k_all.device)
        if mask.dtype != torch.bool or mask.ndim != 2:
            raise ValueError("Current Video selection requires the 2D boolean mixed mask.")
        action_video = mask[action_slice, video_slice]
        visible = action_video[0]
        if not bool(visible.any()) or not torch.equal(action_video, visible.expand_as(action_video)):
            raise ValueError("All Action queries must see the same nonempty current Video frame.")
        count = int(visible.sum())
        if not torch.equal(visible, torch.arange(visible.numel(), device=visible.device) < count):
            raise ValueError("Current Video keys must be a frame-zero prefix of the Video slice.")
        start = int(video_slice.start or 0)
        current_slice = slice(start, start + count)
        other = torch.ones(mask.shape[-1], dtype=torch.bool, device=mask.device)
        other[current_slice] = False
        if bool(mask[current_slice, :][:, other].any()):
            raise ValueError("Current Video must not read future Video, Dream or Action; use first_frame_causal.")
        return k_all[:, current_slice]

    @staticmethod
    def _add_dream_log_gate(scores, allowed, gate, dream_slice):
        """exp(S + log(g)) = g * exp(S): a pre-softmax group attention prior."""
        allowed = allowed.clone()
        allowed[..., dream_slice] &= (gate > 0.0)[:, None, None, :]
        scores = scores.masked_fill(~allowed, -torch.inf)
        # A fresh additive tensor preserves the gate gradient without modifying
        # the masked-fill output in place. Exact zero is a hard mask; log(1)=0.
        log_bias = scores.new_zeros((scores.shape[0], 1, 1, scores.shape[-1]))
        log_bias[..., dream_slice] = torch.log(gate.clamp_min(_LOG_EPS))[:, None, None, :]
        return scores + log_bias

    def _routed_action_attention(self, **kwargs):
        # Keep the learned prior and attention arithmetic in FP32 under AMP.
        # Threshold keeps its existing arithmetic and numerical behavior.
        learned = self.router.config.mode == "learned"
        precision = (torch.autocast(device_type=kwargs["q_action"].device.type, enabled=False)
                     if learned else nullcontext())
        with precision:
            return self._routed_action_attention_impl(**kwargs)

    def _routed_action_attention_impl(
        self,
        *,
        q_action: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        attention_mask: torch.Tensor,
        action_slice: slice,
        dream_slice: slice,
        video_slice: slice,
        layer_idx: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Explicit action-row attention with a multiplicative gate on Dream keys.

        The gate enters as ``+log(gate)`` on the pre-softmax logits, which scales
        the unnormalised attention weight by exactly ``gate``.  A gate of 1 adds
        ``log(1) == 0`` and therefore reproduces the dense distribution; a
        gate of 0 is applied as a mask instead, to avoid ``log(0)`` arithmetic.
        """
        batch_size, action_len, inner_dim = q_action.shape
        total_seq_len = int(k_all.shape[1])
        expected_inner = self.num_heads * self.attn_head_dim
        if inner_dim != expected_inner or k_all.shape[-1] != expected_inner:
            raise ValueError(
                f"Routed attention expected inner dim {expected_inner}, "
                f"got q={inner_dim}, k={k_all.shape[-1]}."
            )

        q = q_action.reshape(batch_size, action_len, self.num_heads, self.attn_head_dim).transpose(1, 2)
        k = k_all.reshape(batch_size, total_seq_len, self.num_heads, self.attn_head_dim).transpose(1, 2)
        v = v_all.reshape(batch_size, total_seq_len, self.num_heads, self.attn_head_dim).transpose(1, 2)

        allowed = self._broadcast_action_mask(
            attention_mask,
            action_slice,
            batch_size=batch_size,
            num_heads=self.num_heads,
            total_seq_len=total_seq_len,
            device=q.device,
        )

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        scores = scores * (self.attn_head_dim ** -0.5)

        need_dense_probs = self.router is not None and self.router.config.mode == "threshold"
        dense_probs = None
        n_valid = None
        if need_dense_probs:
            dense_scores = scores.masked_fill(~allowed, -torch.inf)
            dense_probs = torch.softmax(dense_scores, dim=-1)
            if not bool(torch.isfinite(dense_probs).all()):
                raise FloatingPointError("Dense action attention probabilities contain NaN/Inf.")
            valid_counts = allowed.sum(dim=-1)
            if not torch.equal(valid_counts, valid_counts[:, :1].expand_as(valid_counts)):
                raise ValueError(
                    "The uniform-normalised score requires the valid-key count to be "
                    "identical across heads."
                )
            n_valid = valid_counts[:, 0].to(dtype=dense_probs.dtype)

        k_video_current = None
        if self.router.config.mode == "learned" and self.router.config.gate_type == "va":
            k_video_current = self._current_video_keys(
                k_all, attention_mask, action_slice, video_slice
            )
        gate_info = self.router.gate_for_layer(
            layer_idx=layer_idx,
            q_action=q_action,
            k_video_current=k_video_current,
            k_dream=(k_all[:, dream_slice] if self.router.config.mode == "learned"
                     and self.router.config.gate_type == "da" else None),
            dense_probs=dense_probs,
            dream_slice=dream_slice,
            n_valid=n_valid,
        )
        gate = gate_info["gate"].to(dtype=scores.dtype)

        scores = self._add_dream_log_gate(scores, allowed, gate, dream_slice)

        probs = torch.softmax(scores, dim=-1)
        if not bool(torch.isfinite(probs).all()):
            raise FloatingPointError("Routed action attention probabilities contain NaN/Inf.")
        if self.router.config.mode == "learned":
            out = torch.matmul(probs, v.float()).to(v.dtype)
        else:
            out = torch.matmul(probs.to(dtype=v.dtype), v)
        out = out.transpose(1, 2).reshape(batch_size, action_len, inner_dim)
        if self.capture_action_attention:
            with torch.no_grad():
                head_mean = probs.float().mean(dim=1)  # [B, Sa, S]
                self.captured_action_attention.append(
                    {
                        "layer": int(layer_idx),
                        # [B, Sd]: attention each dream key receives, averaged
                        # over heads and summed over the action chunk's queries.
                        "dream_probs": head_mean[..., dream_slice].mean(dim=1).cpu(),
                        "mass_video": head_mean[..., : dream_slice.start].sum(dim=-1).mean(dim=1).cpu(),
                        "mass_dream": head_mean[..., dream_slice].sum(dim=-1).mean(dim=1).cpu(),
                        "mass_action": head_mean[..., action_slice].sum(dim=-1).mean(dim=1).cpu(),
                        "gate": gate_info["gate"].float().cpu(),
                        "keep": gate_info["keep"].cpu(),
                    }
                )
        return out, gate_info

    @torch.no_grad()
    def _compute_action_attention_probs(
        self, q_cat, k_cat, attention_mask, action_slice, *, dream_slice=None, gate=None,
    ):
        """Report the distribution used by attention, including its actual gate."""
        if gate is None:
            return super()._compute_action_attention_probs(
                q_cat, k_cat, attention_mask, action_slice)
        if q_cat.shape[0] != 1:
            raise ValueError("Action attention debug currently supports batch size 1 only.")
        learned = self.router.config.mode == "learned"
        precision = (torch.autocast(device_type=q_cat.device.type, enabled=False)
                     if learned else nullcontext())
        with precision:
            q = q_cat[:, action_slice].reshape(1, -1, self.num_heads, self.attn_head_dim).transpose(1, 2)
            k = k_cat.reshape(1, -1, self.num_heads, self.attn_head_dim).transpose(1, 2)
            scores = (q.float() @ k.float().transpose(-2, -1)) * (self.attn_head_dim ** -0.5)
            allowed = self._broadcast_action_mask(
                attention_mask, action_slice, batch_size=1, num_heads=self.num_heads,
                total_seq_len=k_cat.shape[1], device=q.device,
            )
            scores = self._add_dream_log_gate(scores, allowed, gate.to(scores.dtype), dream_slice)
            return scores.softmax(dim=-1).cpu()

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        return_action_attention: bool = False,
        attention_layers: Optional[list[int]] = None,
        return_router_statistics: bool = False,
    ):
        # Reset before the early return too: a stale gate list from a previous
        # forward must never reach `budget_loss`.
        self.last_gates = []
        if not self.routing_enabled and not self.capture_kv:
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
        if attention_mask.ndim != 2:
            raise ValueError("RoutedMoT.forward expects a 2D mixed attention mask.")

        self.captured_kv = []
        tokens_all = dict(embeds_all)
        attention_layers_set = None if attention_layers is None else {int(x) for x in attention_layers}
        action_attention_records: list[dict[str, Any]] = []

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
            video_slice = slices["video"]
            total_seq = int(q_cat.shape[1])
            if attention_mask.shape != (total_seq, total_seq):
                raise ValueError("Attention mask does not match the dynamic expert lengths.")
            if action_slice.stop != total_seq:
                raise ValueError("RoutedMoT requires the action expert to be the final chunk.")

            if self.capture_kv:
                self.captured_kv.append(
                    {"k": k_cat[:, dream_slice], "v": v_cat[:, dream_slice]}
                )

            if self.routing_enabled:
                non_action_end = int(action_slice.start)
                non_action_out = flash_attention(
                    q=q_cat[:, :non_action_end],
                    k=k_cat,
                    v=v_cat,
                    num_heads=self.num_heads,
                    ctx_mask=attention_mask[:non_action_end, :],
                )

                def _action_fn(q_action, k_all, v_all, _layer=layer_idx,
                               _action=action_slice, _dream=dream_slice, _video=video_slice):
                    return self._routed_action_attention(
                        q_action=q_action,
                        k_all=k_all,
                        v_all=v_all,
                        attention_mask=attention_mask,
                        action_slice=_action,
                        dream_slice=_dream,
                        video_slice=_video,
                        layer_idx=_layer,
                    )

                if self.mot_checkpoint_mixed_attn and self.training:
                    action_out, gate_info = torch.utils.checkpoint.checkpoint(
                        _action_fn,
                        q_cat[:, action_slice],
                        k_cat,
                        v_cat,
                        use_reentrant=False,
                    )
                else:
                    action_out, gate_info = _action_fn(q_cat[:, action_slice], k_cat, v_cat)
                self._record_gate_info(layer_idx, gate_info)
                mixed = torch.cat([non_action_out, action_out], dim=1)
            else:
                mixed = self._mixed_attention(
                    q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask
                )

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
                            dream_slice=dream_slice,
                            gate=gate_info["gate"] if self.routing_enabled else None,
                        ),
                        "slices": {n: [s.start, s.stop] for n, s in slices.items()},
                    }
                )

            for name in self.expert_order:
                item = cached[name]
                tokens_all[name] = self._apply_post_with_optional_checkpoint(
                    block=item["block"],
                    residual_x=item["residual_x"],
                    gate_msa=item["gate_msa"],
                    shift_mlp=item["shift_mlp"],
                    scale_mlp=item["scale_mlp"],
                    gate_mlp=item["gate_mlp"],
                    use_gradient_checkpointing=item["use_gradient_checkpointing"],
                    mixed_slice=mixed[:, slices[name]],
                    context_payload=context_all.get(name),
                )

        if return_action_attention or return_router_statistics:
            result: dict[str, Any] = {"tokens": tokens_all}
            if return_action_attention:
                result["action_attention"] = action_attention_records
            if return_router_statistics and self.router is not None:
                result["router"] = list(self.router.last_statistics)
            return result
        return tokens_all

    # ------------------------------------------------------ cached inference
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
        dream_token_gates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature_router = getattr(self, "feature_router", None)
        if feature_router is not None and (not feature_router.full or dream_token_gates is not None):
            if dream_token_gates is None:
                raise ValueError("Semantic QK attention requires cached Dream token gates.")
            dream_slice = context_slices["dream"]
            if dream_token_gates.shape != (q_action.shape[0], dream_slice.stop - dream_slice.start):
                raise ValueError("Dream QK gates must be [batch, dream_tokens].")
            mask = attention_mask[action_slice, :].to(device=q_action.device)
            if mask.dtype != torch.bool:
                raise TypeError("Semantic QK routing requires a boolean attention mask.")

            def attend(q, k, v, gates):
                with torch.autocast(device_type=q.device.type, enabled=False):
                    b = q.shape[0]
                    def heads(x):
                        return x.float().reshape(b, -1, self.num_heads, self.attn_head_dim).transpose(1, 2)
                    scores = (heads(q) @ heads(k).transpose(-2, -1)) * (self.attn_head_dim ** -0.5)
                    scores = torch.cat([
                        scores[..., :dream_slice.start],
                        scores[..., dream_slice] * gates.float()[:, None, None, :],
                        scores[..., dream_slice.stop:],
                    ], dim=-1)
                    # Apply masks AFTER scaling: zero gates must never create 0 * -inf.
                    scores = scores.masked_fill(~mask, -torch.inf)
                    # Match SDPA's zero output for a fully masked query row.
                    scores = scores.masked_fill(~mask.any(dim=-1, keepdim=True), 0)
                    probabilities = scores.softmax(dim=-1).masked_fill(~mask, 0)
                    output = probabilities @ heads(v)
                    return output.transpose(1, 2).reshape(b, q.shape[1], -1).to(q.dtype)

            if self.mot_checkpoint_mixed_attn and self.training:
                return torch.utils.checkpoint.checkpoint(
                    attend, q_action, k_all, v_all, dream_token_gates, use_reentrant=False)
            return attend(q_action, k_all, v_all, dream_token_gates)
        if not self.routing_enabled:
            return super()._action_attention_with_context_cache(
                q_action=q_action,
                k_all=k_all,
                v_all=v_all,
                attention_mask=attention_mask,
                action_slice=action_slice,
                context_slices=context_slices,
                layer_idx=layer_idx,
            )
        out, gate_info = self._routed_action_attention(
            q_action=q_action,
            k_all=k_all,
            v_all=v_all,
            attention_mask=attention_mask,
            action_slice=action_slice,
            dream_slice=context_slices["dream"],
            layer_idx=layer_idx,
        )
        # The split training path (Video-once / Dream / Action-cached) runs
        # through here, not through `forward`. Collecting the gate is what makes
        # `ImaginationRouter.budget_loss` see anything: without it the budget
        # term silently evaluates to zero, the router gets no pruning pressure,
        # and the gate stays pinned at its `sigmoid(bias_init)` initialisation.
        self.last_gates.append(gate_info["gate"])
        if self.router is not None:
            self.router.record(
                layer_idx=layer_idx,
                gate=gate_info["gate"],
                keep=gate_info["keep"],
                score=gate_info["score"],
            )

        if self.mot_checkpoint_mixed_attn and self.training:
            out, gate_info = torch.utils.checkpoint.checkpoint(
                attend, q_action, k_all, v_all, use_reentrant=False)
        else:
            out, gate_info = attend(q_action, k_all, v_all)
        # Stage-1 training uses this cached path rather than forward(). Keep
        # gates from exactly this Action forward; logging stays outside recompute.
        self._record_gate_info(layer_idx, gate_info)
        return out

    def forward_action_with_context_cache(self, *args, **kwargs) -> torch.Tensor:
        # Both the statistics and the collected gates belong to one forward;
        # stale gates from a previous call would corrupt the budget loss.
        self.last_gates = []
        if self.router is not None:
            self.router.reset_statistics()
        return super().forward_action_with_context_cache(*args, **kwargs)

    # --------------------------------------------- split Video/Dream prefill
    def forward_dream_with_video_cache(
        self,
        *,
        dream_tokens: torch.Tensor,
        dream_freqs: torch.Tensor,
        dream_t_mod: torch.Tensor,
        dream_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        context_attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> dict[str, Any]:
        """Advance the Dream expert against an already-computed Video K/V cache.

        The Video expert is action-independent *and*, in the single-clean-frame
        setting used at inference, diffusion-step-independent.  Caching it and
        re-running only Dream turns an N-step generative Dream from N full world
        forwards into one Video forward plus N Dream forwards.

        Returns the advanced Dream tokens and the per-layer Dream K/V, which is
        exactly the tensor the action expert will later read -- i.e. the
        "interface" that :mod:`interface_distill` supervises.
        """
        if "dream" not in self.mixtures:
            raise ValueError("RoutedMoT requires a 'dream' expert for the split prefill.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )

        video_seq_len = int(video_seq_len)
        dream_seq_len = int(dream_tokens.shape[1])
        context_seq_len = video_seq_len + dream_seq_len
        if context_attention_mask.ndim != 2 or tuple(context_attention_mask.shape) != (
            context_seq_len,
            context_seq_len,
        ):
            raise ValueError(
                "`context_attention_mask` must have shape "
                f"[{context_seq_len},{context_seq_len}], got {tuple(context_attention_mask.shape)}."
            )

        dream_rows = context_attention_mask[video_seq_len:, :]
        expert = self.mixtures["dream"]
        x = dream_tokens
        dream_kv: list[dict[str, torch.Tensor]] = []

        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            io = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=dream_freqs,
                t_mod=dream_t_mod,
            )
            q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gc = io
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`.")
            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=torch.cat([k_video, k], dim=1),
                v_cat=torch.cat([v_video, v], dim=1),
                attention_mask=dream_rows,
            )
            dream_kv.append({"k": k, "v": v})
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gc,
                mixed_slice=mixed,
                context_payload=dream_context_payload,
            )
        return {"tokens": x, "dream_kv": dream_kv}

    @staticmethod
    def merge_context_cache(
        video_kv_cache: list[dict[str, torch.Tensor]],
        dream_kv_cache: list[dict[str, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        """Concatenate Video and Dream K/V into the layout the action path wants."""
        if len(video_kv_cache) != len(dream_kv_cache):
            raise ValueError(
                f"Cache length mismatch: video={len(video_kv_cache)} dream={len(dream_kv_cache)}."
            )
        return [
            {
                "k": torch.cat([video["k"], dream["k"]], dim=1),
                "v": torch.cat([video["v"], dream["v"]], dim=1),
            }
            for video, dream in zip(video_kv_cache, dream_kv_cache)
        ]
