"""Tests for learning rate scheduler (min_lr_ratio, lr_final)."""

import torch

from tajalli.training.scheduler import get_cosine_warmup_scheduler


def test_scheduler_min_lr_ratio_final_multiplier():
    """With min_lr_ratio=0.1, final step LR should be base_lr * 0.1."""
    base_lr = 6e-4
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=base_lr)
    scheduler = get_cosine_warmup_scheduler(
        optimizer, warmup_steps=100, total_steps=1000, min_lr_ratio=0.1
    )
    for _ in range(999):
        optimizer.zero_grad(set_to_none=True)
        param.grad = torch.zeros_like(param)
        optimizer.step()
        scheduler.step()
    # Last step: step=999; warmup done; progress=1; decay=0; lr = 0.1 + 0.9*0 = 0.1
    actual = optimizer.param_groups[0]["lr"]
    expected = base_lr * 0.1
    assert abs(actual - expected) < 1e-7, f"Final LR {actual} != {expected}"


def test_scheduler_lr_final_derived():
    """When using lr_final derived: min_ratio = lr_final / lr should give correct final LR."""
    base_lr = 6e-4
    lr_final = 1e-5
    min_ratio = lr_final / base_lr  # ~0.01667
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=base_lr)
    scheduler = get_cosine_warmup_scheduler(
        optimizer, warmup_steps=100, total_steps=1000, min_lr_ratio=min_ratio
    )
    for _ in range(999):
        optimizer.zero_grad(set_to_none=True)
        param.grad = torch.zeros_like(param)
        optimizer.step()
        scheduler.step()
    actual = optimizer.param_groups[0]["lr"]
    assert abs(actual - lr_final) < 1e-7, f"Final LR {actual} != {lr_final}"
