"""Loss functions for language and action heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionJsonLoss(nn.Module):
    """L1 + cross-entropy hybrid loss for structured outputs."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight * self.l1(pred, target)
