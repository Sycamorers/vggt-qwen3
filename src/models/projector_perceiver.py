"""Perceiver-style resampler for converting VGGT tokens into Qwen3 space."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@dataclass
class PerceiverConfig:
    latent_dim: int = 4096
    num_latents: int = 128
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 16384
    dropout: float = 0.1


class PerceiverLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(latents, context, context)
        latents = latents + self.dropout(attn_out)
        latents = self.norm1(latents)
        mlp_out = self.mlp(latents)
        latents = self.norm2(latents + self.dropout(mlp_out))
        return latents


class PerceiverProjector(nn.Module):
    """Resample VGGT aggregated tokens to fixed-length latents."""

    def __init__(self, config: PerceiverConfig, in_dim: int, out_dim: int) -> None:
        super().__init__()
        # Initialize latents with proper scaling (std=0.02 is common for transformers)
        self.latents = nn.Parameter(torch.randn(config.num_latents, config.latent_dim) * 0.02)
        self.in_proj = nn.Linear(in_dim, config.latent_dim)
        self.layers = nn.ModuleList(
            [
                PerceiverLayer(config.latent_dim, config.num_heads, config.ffn_dim, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.out_proj = nn.Linear(config.latent_dim, out_dim)
        self.apply(_init_weights)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, T, C] VGGT aggregated tokens.
        Returns:
            latents: [B, num_latents, out_dim] ready for Qwen3.
        """
        B = tokens.size(0)
        context = self.in_proj(tokens)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            latents = layer(latents, context)
        return self.out_proj(latents)
