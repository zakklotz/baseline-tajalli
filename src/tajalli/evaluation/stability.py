"""Stability metrics: drift, hidden norms, essence anchoring."""

import torch
import torch.nn.functional as F


def compute_drift_cosine_sim(h_new: torch.Tensor, h_prev: torch.Tensor) -> float:
    """Cosine similarity between consecutive hidden states (drift metric)."""
    return F.cosine_similarity(
        h_new.flatten(1), h_prev.flatten(1), dim=1
    ).mean().item()


def compute_hidden_norm(h: torch.Tensor) -> float:
    """Mean L2 norm of hidden state across batch and sequence."""
    return h.norm(dim=-1).mean().item()


def compute_essence_anchoring(h: torch.Tensor, tajalli_signal: torch.Tensor) -> float:
    """Cosine similarity between h and tajalli_signal."""
    return F.cosine_similarity(
        h.flatten(1), tajalli_signal.flatten(1), dim=1
    ).mean().item()


def compute_stability_metrics(
    h_steps: list[torch.Tensor],
    tajalli_signals: list[torch.Tensor] | None = None,
) -> dict:
    """
    Compute per-step stability metrics.
    h_steps: list of (B, T, d) tensors from each recursive step
    tajalli_signals: optional, for essence_anchoring (Tajallī model only)
    """
    metrics = {}
    for i, h in enumerate(h_steps):
        metrics[f"step_{i}_hidden_norm"] = compute_hidden_norm(h)
        if i > 0:
            metrics[f"step_{i}_drift_cosine_sim"] = compute_drift_cosine_sim(
                h, h_steps[i - 1]
            )
        if tajalli_signals is not None and i < len(tajalli_signals):
            metrics[f"step_{i}_essence_anchoring"] = compute_essence_anchoring(
                h, tajalli_signals[i]
            )
    return metrics
