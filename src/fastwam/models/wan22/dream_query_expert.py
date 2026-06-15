from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .wan_video_dit import DiTBlock, precompute_freqs_cis


class SharedBlockList(nn.Module):
    """List-like wrapper that exposes one shared block at every layer index."""

    def __init__(self, block: nn.Module, num_layers: int):
        super().__init__()
        self.block = block
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be > 0, got {self.num_layers}")

    def __len__(self) -> int:
        return self.num_layers

    def __getitem__(self, idx: int) -> nn.Module:
        if not isinstance(idx, int):
            raise TypeError(f"SharedBlockList indices must be int, got {type(idx)}")
        if idx < 0:
            idx += self.num_layers
        if idx < 0 or idx >= self.num_layers:
            raise IndexError(f"SharedBlockList index {idx} out of range for length {self.num_layers}")
        return self.block

    def __iter__(self):
        for _ in range(self.num_layers):
            yield self.block


class DreamQueryExpert(nn.Module):
    """Learnable dream-token expert for DreamFastWAM.

    This expert is deliberately not a diffusion model: it does not consume
    Gaussian noise, does not use action/video diffusion timesteps, and runs a
    single forward pass from learned dream queries.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        n_dyn: int,
        n_depth: int,
        n_dino: int,
        n_sam: int,
        dyn_dim: int,
        depth_dim: int,
        dino_dim: int,
        sam_dim: int,
        use_gradient_checkpointing: bool = False,
        share_blocks: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.n_dyn = int(n_dyn)
        self.n_depth = int(n_depth)
        self.n_dino = int(n_dino)
        self.n_sam = int(n_sam)
        self.dyn_dim = int(dyn_dim)
        self.depth_dim = int(depth_dim)
        self.dino_dim = int(dino_dim)
        self.sam_dim = int(sam_dim)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.share_blocks = bool(share_blocks)

        if self.num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {self.num_heads}")
        if self.attn_head_dim <= 0 or self.attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be positive and even for RoPE, got {self.attn_head_dim}"
            )
        for name in ("n_dyn", "n_depth", "n_dino", "n_sam"):
            if getattr(self, name) <= 0:
                raise ValueError(f"`{name}` must be > 0, got {getattr(self, name)}")

        self.dyn_queries = nn.Parameter(torch.randn(1, self.n_dyn, self.hidden_dim) / self.hidden_dim**0.5)
        self.depth_queries = nn.Parameter(torch.randn(1, self.n_depth, self.hidden_dim) / self.hidden_dim**0.5)
        self.dino_queries = nn.Parameter(torch.randn(1, self.n_dino, self.hidden_dim) / self.hidden_dim**0.5)
        self.sam_queries = nn.Parameter(torch.randn(1, self.n_sam, self.hidden_dim) / self.hidden_dim**0.5)

        # Learned zero-time-equivalent modulation used only to satisfy DiTBlock/MoT.
        self.dream_t_mod = nn.Parameter(torch.zeros(1, 6, self.hidden_dim))
        block_kwargs = dict(
            hidden_dim=self.hidden_dim,
            attn_head_dim=self.attn_head_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            eps=eps,
        )
        if self.share_blocks:
            self.blocks = SharedBlockList(DiTBlock(**block_kwargs), int(num_layers))
        else:
            self.blocks = nn.ModuleList([DiTBlock(**block_kwargs) for _ in range(int(num_layers))])
        self.freqs = precompute_freqs_cis(self.attn_head_dim, end=max(1024, self.num_dream_tokens))

        self.dyn_head = nn.Linear(self.hidden_dim, self.dyn_dim)
        self.depth_head = nn.Linear(self.hidden_dim, self.depth_dim)
        self.dino_head = nn.Linear(self.hidden_dim, self.dino_dim)
        self.sam_head = nn.Linear(self.hidden_dim, self.sam_dim)

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

    def pre_dit(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
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
            "dyn": self.dyn_head(tokens[:, slices["dyn"], :]),
            "depth": self.depth_head(tokens[:, slices["depth"], :]),
            "dino": self.dino_head(tokens[:, slices["dino"], :]),
            "sam": self.sam_head(tokens[:, slices["sam"], :]),
        }
