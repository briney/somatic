"""SwiGLU Feed-Forward Network implementation."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FusedSwiGLUFFN(nn.Module):
    """Memory-efficient SwiGLU FFN with fused gate/up projection."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_ffn = d_ffn

        self.w_gate_up = nn.Linear(d_model, d_ffn * 2, bias=bias)
        self.w_down = nn.Linear(d_ffn, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        hidden = self.dropout(hidden)
        return self.w_down(hidden)
