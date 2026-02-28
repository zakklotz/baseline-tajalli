"""Token-level exit router for adaptive recursion (MoR/MoD-style)."""

import torch
import torch.nn as nn


class ExitRouter(nn.Module):
    """
    Lightweight router that predicts per-token exit score at each recursive step.
    forward(h) -> (scores (B,T,1), exit_mask (B,T,1) bool) with optional capacity constraint.
    """

    def __init__(
        self,
        d_model: int,
        threshold: float = 0.5,
        capacity_fraction: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.threshold = threshold
        self.capacity_fraction = capacity_fraction
        self.router = nn.Linear(d_model, 1)

    def forward(
        self,
        h: torch.Tensor,
        exited_so_far: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, T, d_model) — hidden state after this step
            exited_so_far: (B, T, 1) bool — positions that have already exited
        Returns:
            scores: (B, T, 1) in (0, 1) — exit confidence
            exit_mask: (B, T, 1) bool — positions that exit this step (capacity-constrained)
        """
        B, T, _ = h.shape
        scores = torch.sigmoid(self.router(h))  # (B, T, 1)
        score_flat = scores.squeeze(-1)  # (B, T)
        k = max(1, min(T, int(self.capacity_fraction * T + 0.5)))
        # At most k tokens per batch can exit this step: take top-k by score
        _, topk_idx = torch.topk(score_flat, k, dim=1, largest=True)  # (B, k)
        exit_mask = torch.zeros(B, T, 1, dtype=torch.bool, device=h.device)
        exit_mask.scatter_(1, topk_idx.unsqueeze(-1), True)
        exit_mask = exit_mask & (scores > self.threshold) & (~exited_so_far)
        return scores, exit_mask
