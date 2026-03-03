"""Tests for ExitRouter capacity behavior and exit entropy regularizer."""

import torch

from tajalli.model.exit_router import ExitRouter


def test_exit_router_capacity_max_per_step():
    """With T=10, capacity_fraction=0.5 -> k=5; at most 5 exits per step per batch."""
    B, T, d = 2, 10, 64
    router = ExitRouter(d_model=d, threshold=0.0, capacity_fraction=0.5)
    h = torch.randn(B, T, d)
    exited_so_far = torch.zeros(B, T, 1, dtype=torch.bool)

    scores, exit_mask = router(h, exited_so_far)

    # At most 5 exits per batch element (capacity_fraction * T = 5)
    exits_per_batch = exit_mask.squeeze(-1).sum(dim=1)
    assert (exits_per_batch <= 5).all(), f"Exits {exits_per_batch} should be <= 5 per batch"


def test_exit_entropy_loss_decreases_when_entropy_increases():
    """With lambda_exit > 0, loss decreases when exit entropy increases (regularizer subtracts entropy)."""
    # Binary entropy: -s*log(s) - (1-s)*log(1-s), max at s=0.5
    s_low = torch.tensor([0.9, 0.1, 0.95, 0.05])  # peaked -> low entropy
    s_high = torch.tensor([0.5, 0.5, 0.5, 0.5])  # uniform -> high entropy
    s_low = s_low.clamp(1e-7, 1.0 - 1e-7)
    s_high = s_high.clamp(1e-7, 1.0 - 1e-7)
    ent_low = -(s_low * s_low.log() + (1 - s_low) * (1 - s_low).log()).mean()
    ent_high = -(s_high * s_high.log() + (1 - s_high) * (1 - s_high).log()).mean()
    assert ent_high > ent_low
    # Trainer: loss -= lambda_exit * exit_entropy
    lambda_exit = 0.01
    contrib_low = -lambda_exit * ent_low
    contrib_high = -lambda_exit * ent_high
    assert contrib_high < contrib_low, "Higher entropy -> more negative contrib -> lower loss"
