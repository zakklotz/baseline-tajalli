"""Adaptive softmax output layer for efficient LM training with frequency-sorted vocab."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def load_freq_order(path: str | Path) -> list[int]:
    """Load frequency order (most to least frequent token IDs) from JSON."""
    import json

    with open(path) as f:
        data = json.load(f)
    return data["freq_order"]


class AdaptiveOutput(nn.Module):
    """
    Wraps nn.AdaptiveLogSoftmaxWithLoss for frequency-sorted vocab.
    Labels must be remapped to frequency rank (0 = most frequent) before forward.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        cutoffs: list[int],
        div_value: float = 4.0,
        head_bias: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.cutoffs = cutoffs
        self.asm = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=d_model,
            n_classes=vocab_size,
            cutoffs=cutoffs,
            div_value=div_value,
            head_bias=head_bias,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: (N, d_model) flattened hidden states
            target: (N,) label frequency ranks (0 = most frequent), use -1 for ignore
        Returns:
            output: (N,) log probs per target
            loss: scalar NLL loss (excluding ignored)
        """
        ignore = target < 0
        if ignore.all():
            return torch.zeros_like(target, dtype=hidden.dtype), torch.tensor(0.0, device=hidden.device)
        valid = ~ignore
        h_valid = hidden[valid]
        t_valid = target[valid].long()
        out_asm = self.asm(h_valid, t_valid)
        # Build full output for API compatibility
        output_full = torch.zeros(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
        output_full[valid] = out_asm.output
        n_valid = valid.sum().item()
        loss = out_asm.loss if n_valid > 0 else torch.tensor(0.0, device=hidden.device)
        return output_full, loss

    def log_prob(self, hidden: torch.Tensor) -> torch.Tensor:
        """(N, d_model) -> (N, vocab_size) log probabilities in frequency-sorted order."""
        return self.asm.log_prob(hidden)
