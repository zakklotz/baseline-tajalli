"""Qadr output constraints: post-processing of logits (no parameters).

Repetition penalty, temperature scaling, optional vocabulary mask.
Config-driven; can be toggled off. Used at inference or optionally during eval.
"""

from typing import Optional

import torch


class QadrConstraints(torch.nn.Module):
    """
    Stateless logits post-processing. Apply in forward or at eval only.
    """

    def __init__(
        self,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        enable_repetition_penalty: bool = False,
        enable_temperature: bool = False,
    ):
        super().__init__()
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.enable_repetition_penalty = enable_repetition_penalty
        self.enable_temperature = enable_temperature

    def forward(
        self,
        logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """
        logits: (B, T, vocab_size) or (B*T, vocab_size)
        input_ids: (B, T) — used for repetition penalty (recent token IDs to downweight)
        Returns: modified logits (same shape).
        """
        out = logits
        if self.enable_temperature and self.temperature != 1.0:
            out = out / self.temperature
        if self.enable_repetition_penalty and input_ids is not None and self.repetition_penalty != 1.0:
            out = self._apply_repetition_penalty(out, input_ids)
        return out

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Downweight logits for token IDs that appear in input_ids."""
        prev_ids = input_ids.view(-1).unique()
        if prev_ids.numel() == 0:
            return logits
        out = logits.clone()
        out[..., prev_ids] = out[..., prev_ids] / self.repetition_penalty
        return out
