"""Barzakh bottleneck: compress representation before output head.

Continuous variant only: d_model -> d_barzakh -> d_model.
Uses residual (x + decode(encode(x))) so at init the output is approximately x —
decode is initialized to output near zero, so the learned stack representation
passes through and loss starts near the Phase 2 level (e.g. 5–6).
"""

import torch
import torch.nn as nn


class BarzakhBottleneck(nn.Module):
    """
    Continuous bottleneck between stack output and output head.
    encode: Linear(d_model, d_barzakh) + LayerNorm(d_barzakh)
    decode: Linear(d_barzakh, d_model), init so output is near zero.
    forward: x + decode(encode(x)) — residual so at init, x passes through.
    """

    def __init__(self, d_model: int, d_barzakh: int):
        super().__init__()
        self.d_model = d_model
        self.d_barzakh = d_barzakh
        self.encode = nn.Sequential(
            nn.Linear(d_model, d_barzakh),
            nn.LayerNorm(d_barzakh),
        )
        self.decode = nn.Linear(d_barzakh, d_model)
        # At init, decode outputs ~0 so output = x + 0 ≈ x (no destruction of learned repr)
        nn.init.zeros_(self.decode.bias)
        nn.init.zeros_(self.decode.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model). Residual: x + decode(encode(x))."""
        z = self.encode(x)
        return x + self.decode(z)
