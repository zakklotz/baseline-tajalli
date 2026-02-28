"""Phase 2 auxiliary losses: pair orthogonality, pair coverage, load balance."""

import torch
import torch.nn.functional as F
from typing import Optional


def pair_orthogonality_loss(
    expert_outputs: torch.Tensor,
    margin: float = 0.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Penalize high cosine similarity between paired expert outputs.
    expert_outputs: (N, n_experts, d_model) — outputs of all experts on N tokens.
    For each pair (0,1), (2,3), ... compute mean cosine_sim and push below margin.
    """
    N, n_experts, d_model = expert_outputs.shape
    n_pairs = n_experts // 2
    if mask is not None and mask.dim() == 1:
        mask = mask.to(expert_outputs.device).float()
    elif mask is not None and mask.dim() == 2:
        mask = mask.float().mean(dim=-1)

    losses = []
    for p in range(n_pairs):
        i, j = 2 * p, 2 * p + 1
        ei = expert_outputs[:, i, :]  # (N, d_model)
        ej = expert_outputs[:, j, :]
        cos_sim = F.cosine_similarity(ei.unsqueeze(0), ej.unsqueeze(0), dim=2).squeeze(0)
        if mask is not None:
            cos_sim = cos_sim * mask
            denom = mask.sum().clamp(min=1e-6)
            pair_loss = (cos_sim.sum() / denom - margin).clamp(min=0)
        else:
            pair_loss = (cos_sim.mean() - margin).clamp(min=0)
        losses.append(pair_loss)
    return torch.stack(losses).mean()


def pair_coverage_loss(
    expert_indices: torch.Tensor,
    num_experts: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Penalize imbalance within each pair: ratio = min(freq_i, freq_j) / max(freq_i, freq_j).
    loss = mean over pairs of (1 - ratio).
    expert_indices: (B, T, top_k) token-choice indices, or (B, T, n_experts) expert-choice contribution mask.
    num_experts: if None, inferred from expert_indices when K > 2 (expert-choice), else 14.
    """
    B, T, K = expert_indices.shape
    if num_experts is None:
        num_experts = K if K > 2 else 14
    n_pairs = num_experts // 2
    device = expert_indices.device
    if mask is not None:
        if mask.dim() == 3:
            token_mask = (mask.abs().sum(dim=(1, 2)) > 0).float().reshape(-1)
        elif mask.dim() == 2:
            token_mask = mask.float().reshape(-1)
        else:
            token_mask = torch.ones(B * T, device=device)
    else:
        token_mask = torch.ones(B * T, device=device)

    losses = []
    for p in range(n_pairs):
        i, j = 2 * p, 2 * p + 1
        if K == num_experts:
            # expert-choice: (B, T, n_experts) contribution mask
            count_i = (expert_indices[:, :, i] * token_mask.view(B, T)).sum()
            count_j = (expert_indices[:, :, j] * token_mask.view(B, T)).sum()
        else:
            # token-choice: (B, T, top_k) indices
            flat = expert_indices.reshape(-1, K)
            in_i = (flat == i).any(dim=1).float() * token_mask
            in_j = (flat == j).any(dim=1).float() * token_mask
            count_i = in_i.sum()
            count_j = in_j.sum()
        total = count_i + count_j
        if total < 1e-6:
            ratio = torch.tensor(1.0, device=device, dtype=expert_indices.dtype)
        else:
            ratio = torch.min(count_i, count_j) / torch.max(count_i, count_j).clamp(min=1e-6)
        losses.append(1.0 - ratio)
    return torch.stack(losses).mean()


def load_balance_loss_from_router(load_balance_loss: torch.Tensor) -> torch.Tensor:
    """Use the scalar load_balance_loss produced by WisdomRouter as-is."""
    return load_balance_loss
