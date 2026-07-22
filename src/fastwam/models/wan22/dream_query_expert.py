from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Any, Dict

import torch
import torch.nn as nn

from .wan_video_dit import DiTBlock, precompute_freqs_cis
from fastwam.utils.logging_config import get_logger


logger = get_logger(__name__)

_DREAM_MODALITIES = ("dyn", "depth", "dino", "sam")


class DenseDreamDecoder(nn.Module):
    """Independent modality decoder from dream latent tokens to dense targets."""

    def __init__(
        self,
        *,
        modality: str,
        latent_dim: int,
        target_shape: list[int] | tuple[int, ...],
        target_layout: str = "token_feature",
        decoder_dim: int,
        decoder_ffn_dim: int,
        num_layers: int,
        num_heads: int,
        attn_head_dim: int | None = None,
        enabled: bool = True,
        type: str | None = None,
    ):
        super().__init__()
        self.modality = modality
        self.enabled = bool(enabled)
        self.type = type
        self.target_layout = str(target_layout)
        self.decoder_dim = int(decoder_dim)
        self.decoder_ffn_dim = int(decoder_ffn_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim) if attn_head_dim is not None else None
        if not self.enabled:
            self.target_shape = None
            return
        if self.decoder_dim <= 0:
            raise ValueError(f"dream_decoder.decoder_dim must be > 0, got {self.decoder_dim}")
        if self.decoder_ffn_dim <= 0:
            raise ValueError(f"dream_decoder.decoder_ffn_dim must be > 0, got {self.decoder_ffn_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"dream_decoder.num_heads must be > 0, got {self.num_heads}")
        if self.decoder_dim % self.num_heads != 0:
            raise ValueError(
                f"dream_decoder.decoder_dim must be divisible by num_heads, "
                f"got decoder_dim={self.decoder_dim}, num_heads={self.num_heads}."
            )
        if target_shape is None:
            raise ValueError(
                f"dream_decoder.{modality}.target_shape is null. "
                "Run scripts/discover_dream_target_shapes.py --write-config first."
            )
        self.target_shape = tuple(int(x) for x in target_shape)
        if len(self.target_shape) == 0 or any(x <= 0 for x in self.target_shape):
            raise ValueError(f"Invalid dream_decoder.{modality}.target_shape={target_shape}")

        if self.target_layout == "token_feature":
            if len(self.target_shape) != 2:
                raise ValueError(
                    f"dream_decoder.{modality}.target_layout=token_feature requires "
                    f"target_shape=[num_tokens, feature_dim], got {target_shape}."
                )
            self.num_output_tokens = int(self.target_shape[0])
            self.feature_dim = int(self.target_shape[1])
            self.output_grid_shape = None
        elif self.target_layout == "grid_feature":
            if len(self.target_shape) != 3:
                raise ValueError(
                    f"dream_decoder.{modality}.target_layout=grid_feature requires "
                    f"target_shape=[grid_h, grid_w, feature_dim], got {target_shape}."
                )
            grid_h, grid_w = int(self.target_shape[0]), int(self.target_shape[1])
            self.num_output_tokens = grid_h * grid_w
            self.feature_dim = int(self.target_shape[-1])
            self.output_grid_shape = self.target_shape[:-1]
        elif self.target_layout == "image":
            raise ValueError(
                f"dream_decoder.{modality}.target_layout=image is not supported in the first version. "
                "Patchify image-like targets in the dataset and use token_feature."
            )
        else:
            raise ValueError(
                f"Unsupported dream_decoder.{modality}.target_layout={self.target_layout!r}; "
                "expected token_feature or grid_feature."
            )

        self.latent_proj = nn.Linear(int(latent_dim), self.decoder_dim)
        self.output_queries = nn.Parameter(
            torch.randn(1, self.num_output_tokens, self.decoder_dim) / max(self.decoder_dim, 1) ** 0.5
        )
        layer = nn.TransformerDecoderLayer(
            d_model=self.decoder_dim,
            nhead=self.num_heads,
            dim_feedforward=self.decoder_ffn_dim,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=self.num_layers)
        self.output_proj = nn.Linear(self.decoder_dim, self.feature_dim)

    def forward(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError(f"Decoder for modality={self.modality!r} is disabled.")
        if latent_tokens.ndim != 3:
            raise ValueError(
                f"{self.modality} latent tokens must be [B, N, D], got {tuple(latent_tokens.shape)}"
            )
        memory = self.latent_proj(latent_tokens)
        queries = self.output_queries.to(device=latent_tokens.device, dtype=memory.dtype).expand(
            latent_tokens.shape[0], -1, -1
        )
        decoded = self.decoder(tgt=queries, memory=memory)
        out = self.output_proj(decoded)
        if self.target_layout == "grid_feature":
            return out.reshape(latent_tokens.shape[0], *self.target_shape)
        return out


class DreamQueryExpert(nn.Module):
    """Learnable dream latent expert plus independent dense modality decoders."""

    def __init__(
        self,
        *,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int | None = None,
        hidden_dim: int | None = None,
        ffn_dim: int | None = None,
        dream_query: dict[str, Any] | None = None,
        dream_expert: dict[str, Any] | None = None,
        dream_decoder: dict[str, Any] | None = None,
        use_gradient_checkpointing: bool = False,
        share_blocks: bool = False,
        **legacy_kwargs,
    ):
        super().__init__()
        dream_query = dict(dream_query or {})
        dream_expert = dict(dream_expert or {})
        dream_decoder = dict(dream_decoder or {})

        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.num_heads = int(dream_expert.get("num_heads") or num_heads)
        self.attn_head_dim = int(dream_expert.get("attn_head_dim") or attn_head_dim)
        self.num_layers = int(dream_expert.get("num_layers") or num_layers or 30)
        self.hidden_dim = int(dream_expert.get("hidden_dim") or hidden_dim or 384)
        mlp_ratio = float(dream_expert.get("mlp_ratio", 4.0))
        explicit_ffn_dim = dream_expert.get("ffn_dim", None)
        if explicit_ffn_dim is None:
            fallback_ffn_dim = ffn_dim if ffn_dim is not None else round(self.hidden_dim * mlp_ratio)
            self.ffn_dim = int(fallback_ffn_dim)
            logger.warning(
                "dream_expert.ffn_dim is not configured; resolved ffn_dim=%d from %s.",
                self.ffn_dim,
                "legacy ffn_dim" if ffn_dim is not None else "round(hidden_dim * mlp_ratio)",
            )
        else:
            self.ffn_dim = int(explicit_ffn_dim)
            ratio_ffn_dim = float(self.hidden_dim) * mlp_ratio
            if abs(float(self.ffn_dim) - ratio_ffn_dim) > 1e-6:
                logger.warning(
                    "dream_expert.ffn_dim=%d does not match hidden_dim * mlp_ratio=%.3f "
                    "(hidden_dim=%d, mlp_ratio=%.3f). Using explicit ffn_dim.",
                    self.ffn_dim,
                    ratio_ffn_dim,
                    self.hidden_dim,
                    mlp_ratio,
                )
        self.target_num_params_m = dream_expert.get("target_num_params_m", None)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.share_blocks = bool(dream_expert.get("share_blocks", share_blocks))

        self.n_dyn = int(dream_query.get("n_dyn", legacy_kwargs.get("n_dyn", 8)))
        self.n_depth = int(dream_query.get("n_depth", legacy_kwargs.get("n_depth", 8)))
        self.n_dino = int(dream_query.get("n_dino", legacy_kwargs.get("n_dino", 8)))
        self.n_sam = int(dream_query.get("n_sam", legacy_kwargs.get("n_sam", 8)))
        modalities = dream_query.get("modalities", legacy_kwargs.get("modalities", _DREAM_MODALITIES))
        if isinstance(modalities, str):
            modalities = [modalities]
        self.modalities = tuple(str(name) for name in modalities)
        if not self.modalities:
            raise ValueError("dream_query.modalities must contain at least one modality.")
        unknown_modalities = set(self.modalities) - set(_DREAM_MODALITIES)
        if unknown_modalities:
            raise ValueError(
                f"Unsupported dream_query.modalities: {sorted(unknown_modalities)}. "
                f"Expected subset of {list(_DREAM_MODALITIES)}."
            )
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError(f"dream_query.modalities contains duplicates: {list(self.modalities)}")
        future_offsets = dream_query.get("future_offsets", dream_query.get("future_steps", None))
        if future_offsets is None:
            future_offsets = legacy_kwargs.get("future_offsets", [0])
        if isinstance(future_offsets, int):
            future_offsets = [future_offsets]
        self.future_offsets = [int(x) for x in future_offsets]
        if not self.future_offsets:
            raise ValueError("dream_query.future_offsets must contain at least one offset.")
        self.num_future_offsets = len(self.future_offsets)

        for name in ("hidden_dim", "ffn_dim", "num_layers", "num_heads", "attn_head_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"`dream_expert.{name}` must be > 0, got {getattr(self, name)}")
        if self.attn_head_dim <= 0 or self.attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be positive and even for RoPE, got {self.attn_head_dim}")
        for modality in self.modalities:
            name = f"n_{modality}"
            if getattr(self, name) <= 0:
                raise ValueError(f"`dream_query.{name}` must be > 0, got {getattr(self, name)}")

        for modality in _DREAM_MODALITIES:
            n_tokens = getattr(self, f"n_{modality}")
            param_name = f"{modality}_queries"
            if modality in self.modalities:
                self.register_parameter(
                    param_name,
                    nn.Parameter(
                        torch.randn(self.num_future_offsets, n_tokens, self.hidden_dim) / self.hidden_dim**0.5
                    ),
                )
            else:
                self.register_parameter(param_name, None)

        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.dream_t_mod = nn.Parameter(torch.zeros(1, 6, self.hidden_dim))
        block_kwargs = dict(
            hidden_dim=self.hidden_dim,
            attn_head_dim=self.attn_head_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            eps=eps,
        )
        if self.share_blocks:
            raise ValueError("DreamFastWAM dense decoder setup requires `share_blocks=false` for parameter scale.")
        self.blocks = nn.ModuleList([DiTBlock(**block_kwargs) for _ in range(self.num_layers)])
        self.freqs = precompute_freqs_cis(self.attn_head_dim, end=max(1024, self.num_dream_tokens))

        if dream_decoder.get("decoder_dim", None) is None:
            raise ValueError("dream_decoder.decoder_dim must be explicitly configured.")
        decoder_dim = int(dream_decoder["decoder_dim"])
        decoder_mlp_ratio = float(dream_decoder.get("mlp_ratio", 4.0))
        explicit_decoder_ffn_dim = dream_decoder.get("decoder_ffn_dim", None)
        if explicit_decoder_ffn_dim is None:
            decoder_ffn_dim = int(round(decoder_dim * decoder_mlp_ratio))
            logger.warning(
                "dream_decoder.decoder_ffn_dim is not configured; resolved decoder_ffn_dim=%d "
                "from round(decoder_dim * mlp_ratio).",
                decoder_ffn_dim,
            )
        else:
            decoder_ffn_dim = int(explicit_decoder_ffn_dim)
            ratio_decoder_ffn_dim = float(decoder_dim) * decoder_mlp_ratio
            if abs(float(decoder_ffn_dim) - ratio_decoder_ffn_dim) > 1e-6:
                logger.warning(
                    "dream_decoder.decoder_ffn_dim=%d does not match decoder_dim * mlp_ratio=%.3f "
                    "(decoder_dim=%d, mlp_ratio=%.3f). Using explicit decoder_ffn_dim.",
                    decoder_ffn_dim,
                    ratio_decoder_ffn_dim,
                    decoder_dim,
                    decoder_mlp_ratio,
                )
        decoder_layers = int(dream_decoder.get("num_layers", 2))
        decoder_heads = int(dream_decoder.get("num_heads") or self.num_heads)
        decoder_attn_head_dim = int(dream_decoder.get("attn_head_dim") or self.attn_head_dim)
        for name, value in (
            ("decoder_dim", decoder_dim),
            ("decoder_ffn_dim", decoder_ffn_dim),
            ("num_layers", decoder_layers),
            ("num_heads", decoder_heads),
            ("attn_head_dim", decoder_attn_head_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"`dream_decoder.{name}` must be > 0, got {value}")
        if decoder_dim % decoder_heads != 0:
            raise ValueError(
                f"dream_decoder.decoder_dim must be divisible by num_heads, "
                f"got decoder_dim={decoder_dim}, num_heads={decoder_heads}."
            )
        self.decoder_dim = decoder_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_num_layers = decoder_layers
        self.decoder_num_heads = decoder_heads
        self.decoder_attn_head_dim = decoder_attn_head_dim
        self.decoders = nn.ModuleDict()
        for modality in self.modalities:
            cfg = dict(dream_decoder.get(modality, {}) or {})
            enabled = bool(cfg.get("enabled", True))
            self.decoders[modality] = DenseDreamDecoder(
                modality=modality,
                latent_dim=self.hidden_dim,
                target_shape=cfg.get("target_shape"),
                target_layout=cfg.get("target_layout", "token_feature"),
                decoder_dim=decoder_dim,
                decoder_ffn_dim=decoder_ffn_dim,
                num_layers=decoder_layers,
                num_heads=decoder_heads,
                attn_head_dim=decoder_attn_head_dim,
                enabled=enabled,
                type=cfg.get("type"),
            )

        self.architecture = self._resolved_architecture()

    @property
    def num_dream_tokens(self) -> int:
        return sum(self.num_future_offsets * getattr(self, f"n_{name}") for name in self.modalities)

    def modality_slices(self) -> dict[str, slice]:
        start = 0
        slices = {}
        for name in self.modalities:
            length = self.num_future_offsets * getattr(self, f"n_{name}")
            end = start + length
            slices[name] = slice(start, end)
            start = end
        return slices

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _resolved_architecture(self) -> dict[str, Any]:
        return {
            "dream_expert": {
                "hidden_dim": self.hidden_dim,
                "ffn_dim": self.ffn_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "attn_head_dim": self.attn_head_dim,
                "num_dream_tokens": self.num_dream_tokens,
                "modalities": list(self.modalities),
                "future_offsets": list(self.future_offsets),
                "num_future_offsets": self.num_future_offsets,
                "target_num_params_m": self.target_num_params_m,
            },
            "dream_decoder": {
                "decoder_dim": self.decoder_dim,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "num_layers": self.decoder_num_layers,
                "num_heads": self.decoder_num_heads,
                "attn_head_dim": self.decoder_attn_head_dim,
                "enabled_modalities": list(self.modalities),
                "modalities": {
                    name: {
                        "enabled": bool(decoder.enabled),
                        "target_layout": getattr(decoder, "target_layout", None),
                        "target_shape": list(decoder.target_shape) if getattr(decoder, "target_shape", None) is not None else None,
                    }
                    for name, decoder in self.decoders.items()
                },
            },
        }

    def resolved_architecture(self) -> dict[str, Any]:
        return self.architecture

    def pre_dit(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, got {batch_size}")
        query_chunks = []
        for modality in self.modalities:
            query = getattr(self, f"{modality}_queries")
            if query is None:
                raise RuntimeError(f"Dream modality {modality!r} is enabled but has no query parameter.")
            query_chunks.append(
                query.reshape(
                    1,
                    self.num_future_offsets * getattr(self, f"n_{modality}"),
                    self.hidden_dim,
                )
            )
        queries = torch.cat(query_chunks, dim=1).to(device=device, dtype=dtype)
        tokens = queries.expand(batch_size, -1, -1).contiguous()
        freqs = self.freqs[: self.num_dream_tokens].view(self.num_dream_tokens, 1, -1).to(tokens.device)
        t_mod = self.dream_t_mod.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        pre_state = {
            "tokens": tokens,
            "freqs": freqs,
            "t_mod": t_mod,
            "meta": {
                "batch_size": batch_size,
                "seq_len": self.num_dream_tokens,
                "modality_slices": self.modality_slices(),
                "num_future_offsets": self.num_future_offsets,
                "future_offsets": list(self.future_offsets),
            },
        }
        if context is not None:
            if context.ndim != 3:
                raise ValueError(f"`context` must be 3D [B,L,D], got shape {tuple(context.shape)}")
            if context.shape[0] != batch_size:
                raise ValueError(
                    f"Batch mismatch between dream tokens and text context: {batch_size} vs {context.shape[0]}"
                )
            if context_mask is None:
                context_mask = torch.ones((batch_size, context.shape[1]), dtype=torch.bool, device=context.device)
            else:
                if context_mask.ndim != 2:
                    raise ValueError(f"`context_mask` must be 2D [B,L], got shape {tuple(context_mask.shape)}")
                if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                    raise ValueError(
                        f"`context_mask` shape must match `context` shape [B,L], "
                        f"got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                    )
            context = context.to(device=device, dtype=dtype)
            context_mask = context_mask.to(device=device, dtype=torch.bool)
            pre_state["context"] = self.text_embedding(context)
            pre_state["context_mask"] = context_mask.unsqueeze(1).expand(-1, self.num_dream_tokens, -1)
        else:
            pre_state["context"] = None
            pre_state["context_mask"] = None
        return pre_state

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"`tokens` must be 3D [B, S, D], got shape {tuple(tokens.shape)}")
        expected = self.num_dream_tokens
        if tokens.shape[1] != expected:
            raise ValueError(f"Dream token length mismatch: expected {expected}, got {tokens.shape[1]}")
        slices = pre_state["meta"]["modality_slices"]
        out = {}
        for name in self.modalities:
            decoder = self.decoders[name]
            if not bool(getattr(decoder, "enabled", True)):
                continue
            modality_tokens = tokens[:, slices[name], :]
            bsz, _, dim = modality_tokens.shape
            per_offset_tokens = getattr(self, f"n_{name}")
            modality_tokens = modality_tokens.reshape(
                bsz, self.num_future_offsets, per_offset_tokens, dim
            )
            decoded = decoder(modality_tokens.reshape(bsz * self.num_future_offsets, per_offset_tokens, dim))
            out[name] = decoded.reshape(bsz, self.num_future_offsets, *decoded.shape[1:])
        return out
