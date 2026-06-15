from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Any, Dict

import torch
import torch.nn as nn

from .wan_video_dit import DiTBlock, precompute_freqs_cis


class DenseDreamDecoder(nn.Module):
    """Independent modality decoder from dream latent tokens to dense targets."""

    def __init__(
        self,
        *,
        modality: str,
        latent_dim: int,
        target_shape: list[int] | tuple[int, ...],
        decoder_dim: int,
        num_layers: int,
        num_heads: int,
        enabled: bool = True,
        type: str | None = None,
    ):
        super().__init__()
        self.modality = modality
        self.enabled = bool(enabled)
        self.type = type
        if not self.enabled:
            self.target_shape = None
            return
        if target_shape is None:
            raise ValueError(
                f"dream_decoder.{modality}.target_shape is required. "
                "Run `python scripts/discover_dream_target_shapes.py --write-config ...` first."
            )
        self.target_shape = tuple(int(x) for x in target_shape)
        if len(self.target_shape) == 0 or any(x <= 0 for x in self.target_shape):
            raise ValueError(f"Invalid dream_decoder.{modality}.target_shape={target_shape}")

        if modality in {"dino", "sam"}:
            if len(self.target_shape) < 2:
                raise ValueError(f"{modality} target_shape must include a feature dimension, got {target_shape}")
            self.feature_dim = int(self.target_shape[-1])
            self.num_output_tokens = int(reduce(mul, self.target_shape[:-1], 1))
            self.output_grid_shape = self.target_shape[:-1]
        else:
            self.feature_dim = 1
            self.num_output_tokens = int(reduce(mul, self.target_shape, 1))
            self.output_grid_shape = self.target_shape

        self.latent_proj = nn.Linear(int(latent_dim), int(decoder_dim))
        self.output_queries = nn.Parameter(
            torch.randn(1, self.num_output_tokens, int(decoder_dim)) / max(int(decoder_dim), 1) ** 0.5
        )
        layer = nn.TransformerDecoderLayer(
            d_model=int(decoder_dim),
            nhead=int(num_heads),
            dim_feedforward=int(decoder_dim) * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(num_layers))
        self.output_proj = nn.Linear(int(decoder_dim), self.feature_dim)

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
        if self.feature_dim == 1:
            return out.squeeze(-1).reshape(latent_tokens.shape[0], *self.target_shape)
        return out.reshape(latent_tokens.shape[0], *self.target_shape)


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
        self.attn_head_dim = int(attn_head_dim)
        self.num_layers = int(dream_expert.get("num_layers") or num_layers or 30)
        self.hidden_dim = int(dream_expert.get("hidden_dim") or hidden_dim or 384)
        mlp_ratio = float(dream_expert.get("mlp_ratio", 4.0))
        self.ffn_dim = int(dream_expert.get("ffn_dim") or ffn_dim or round(self.hidden_dim * mlp_ratio))
        self.target_num_params_m = dream_expert.get("target_num_params_m", None)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.share_blocks = bool(share_blocks)

        self.n_dyn = int(dream_query.get("n_dyn", legacy_kwargs.get("n_dyn", 8)))
        self.n_depth = int(dream_query.get("n_depth", legacy_kwargs.get("n_depth", 8)))
        self.n_dino = int(dream_query.get("n_dino", legacy_kwargs.get("n_dino", 8)))
        self.n_sam = int(dream_query.get("n_sam", legacy_kwargs.get("n_sam", 8)))

        if self.num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {self.num_heads}")
        if self.attn_head_dim <= 0 or self.attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be positive and even for RoPE, got {self.attn_head_dim}")
        for name in ("n_dyn", "n_depth", "n_dino", "n_sam"):
            if getattr(self, name) <= 0:
                raise ValueError(f"`dream_query.{name}` must be > 0, got {getattr(self, name)}")

        self.dyn_queries = nn.Parameter(torch.randn(1, self.n_dyn, self.hidden_dim) / self.hidden_dim**0.5)
        self.depth_queries = nn.Parameter(torch.randn(1, self.n_depth, self.hidden_dim) / self.hidden_dim**0.5)
        self.dino_queries = nn.Parameter(torch.randn(1, self.n_dino, self.hidden_dim) / self.hidden_dim**0.5)
        self.sam_queries = nn.Parameter(torch.randn(1, self.n_sam, self.hidden_dim) / self.hidden_dim**0.5)

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

        decoder_dim = int(dream_decoder.get("decoder_dim") or self.hidden_dim)
        decoder_layers = int(dream_decoder.get("num_layers", 2))
        decoder_heads = int(dream_decoder.get("num_heads") or self.num_heads)
        self.decoders = nn.ModuleDict()
        for modality in ("dyn", "depth", "dino", "sam"):
            cfg = dict(dream_decoder.get(modality, {}) or {})
            enabled = bool(cfg.get("enabled", True))
            self.decoders[modality] = DenseDreamDecoder(
                modality=modality,
                latent_dim=self.hidden_dim,
                target_shape=cfg.get("target_shape"),
                decoder_dim=decoder_dim,
                num_layers=decoder_layers,
                num_heads=decoder_heads,
                enabled=enabled,
                type=cfg.get("type"),
            )

    @property
    def num_dream_tokens(self) -> int:
        return self.n_dyn + self.n_depth + self.n_dino + self.n_sam

    def modality_slices(self) -> dict[str, slice]:
        start = 0
        slices = {}
        for name, length in (
            ("dyn", self.n_dyn),
            ("depth", self.n_depth),
            ("dino", self.n_dino),
            ("sam", self.n_sam),
        ):
            end = start + length
            slices[name] = slice(start, end)
            start = end
        return slices

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def pre_dit(self, batch_size: int, device: torch.device | str, dtype: torch.dtype) -> Dict[str, Any]:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, got {batch_size}")
        queries = torch.cat(
            [self.dyn_queries, self.depth_queries, self.dino_queries, self.sam_queries],
            dim=1,
        ).to(device=device, dtype=dtype)
        tokens = queries.expand(batch_size, -1, -1).contiguous()
        freqs = self.freqs[: self.num_dream_tokens].view(self.num_dream_tokens, 1, -1).to(tokens.device)
        t_mod = self.dream_t_mod.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t_mod": t_mod,
            "meta": {
                "batch_size": batch_size,
                "seq_len": self.num_dream_tokens,
                "modality_slices": self.modality_slices(),
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"`tokens` must be 3D [B, S, D], got shape {tuple(tokens.shape)}")
        expected = self.num_dream_tokens
        if tokens.shape[1] != expected:
            raise ValueError(f"Dream token length mismatch: expected {expected}, got {tokens.shape[1]}")
        slices = pre_state["meta"]["modality_slices"]
        return {
            name: self.decoders[name](tokens[:, slices[name], :])
            for name in ("dyn", "depth", "dino", "sam")
            if bool(getattr(self.decoders[name], "enabled", True))
        }
