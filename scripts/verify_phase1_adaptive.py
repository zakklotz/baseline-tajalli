#!/usr/bin/env python3
"""Verify Phase 1 Adaptive mechanics: essence shapes, depth families, alpha, inner layers, gate/exit entropy, RoPE+SDPA."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure tajalli and nn-core are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

# 1) Matrix essence + matrix attribute heads
def check_matrix_essence_and_heads() -> None:
    from tajalli.model.tajalli_model import TajalliModelPhase1
    from tajalli.model.init import init_tajalli_weights
    from tajalli.model.tajalli import AttributeHeadSimple

    B, T, d_model, d_essence, n_rows = 2, 8, 768, 1536, 64
    model = TajalliModelPhase1(
        vocab_size=50257,
        d_model=d_model,
        d_essence=d_essence,
        n_heads=12,
        d_head=64,
        d_ff=3072,
        n_steps=8,
        essence_type="matrix",
        n_essence_rows=n_rows,
        alpha_schedule=[0.25] * 8,
        n_inner_layers=3,
        use_exit_router=True,
        use_recursive_kv_cache=True,
        depth_families=3,
        family_steps=[0, 3, 6],
    )
    init_tajalli_weights(model, n_steps=8)

    # Essence shape: (B, 64, d_model)
    essence = model.essence(B)
    assert essence.shape == (B, 64, d_model), f"essence.shape={essence.shape}, expected (B, 64, {d_model})"

    # Row logits shape (B, T, 64) from AttributeHeadSimple
    head = AttributeHeadSimple(d_model=d_model, n_essence_rows=n_rows)
    h = torch.randn(B, T, d_model)
    row_logits = head.row_weights(h)
    assert row_logits.shape == (B, T, 64), f"row_logits.shape={row_logits.shape}, expected (B, T, 64)"


# 2) Depth families param names
def check_depth_families() -> None:
    from tajalli.model.tajalli_model import TajalliModelPhase1
    from tajalli.model.init import init_tajalli_weights

    model = TajalliModelPhase1(
        vocab_size=50257,
        d_model=64,
        d_essence=64,
        n_heads=4,
        d_head=16,
        d_ff=128,
        n_steps=8,
        essence_type="matrix",
        n_essence_rows=32,
        depth_families=3,
        family_steps=[0, 3, 6],
    )
    init_tajalli_weights(model, n_steps=8)

    names = {n for n, _ in model.named_parameters()}
    assert "tajalli_stack.block.tajalli_layer.attribute_heads_family.0.0.row_weights.weight" in names
    assert "tajalli_stack.block.tajalli_layer.attribute_heads_family.1.0.row_weights.weight" in names
    assert "tajalli_stack.block.tajalli_layer.gate_family.0.weight" in names
    assert "tajalli_stack.block.tajalli_layer.gate_family.1.weight" in names


# 3) Alpha schedule init
def check_alpha_schedule() -> None:
    from tajalli.model.tajalli_model import TajalliModelPhase1
    from tajalli.model.init import init_tajalli_weights

    model = TajalliModelPhase1(
        vocab_size=50257,
        d_model=64,
        d_essence=64,
        n_heads=4,
        d_head=16,
        d_ff=128,
        n_steps=8,
        essence_type="vector",
        alpha_schedule=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    )
    init_tajalli_weights(model, n_steps=8)

    alpha0 = torch.sigmoid(model.tajalli_stack.block.tajalli_layer.alpha_per_step[0]).item()
    assert abs(alpha0 - 0.25) < 0.01, f"alpha[0]={alpha0}, expected ~0.25"


# 4) Inner layers
def check_inner_layers() -> None:
    from tajalli.model.tajalli_model import TajalliModelPhase1
    from tajalli.model.init import init_tajalli_weights

    model = TajalliModelPhase1(
        vocab_size=50257,
        d_model=64,
        d_essence=64,
        n_heads=4,
        d_head=16,
        d_ff=128,
        n_steps=8,
        n_inner_layers=3,
    )
    init_tajalli_weights(model, n_steps=8)

    block = model.tajalli_stack.block
    assert len(block.inner_attns) == 3, f"inner_attns length={len(block.inner_attns)}"
    assert len(block.inner_ffns) == 3, f"inner_ffns length={len(block.inner_ffns)}"


# 5) Gate entropy + exit entropy in loss (wiring check)
def check_gate_exit_entropy_wiring() -> None:
    from tajalli.model.tajalli import TajalliStack
    from tajalli.model.essence import EssenceCoreMatrix

    B, T, d_model = 2, 4, 64
    essence = EssenceCoreMatrix(n_rows=32, d_model=d_model)
    stack = TajalliStack(
        d_model=d_model,
        d_essence=d_model,
        n_heads=4,
        d_head=16,
        d_ff=128,
        n_steps=2,
        essence_type="matrix",
        n_essence_rows=32,
        exit_router=__import__("tajalli.model.exit_router", fromlist=["ExitRouter"]).ExitRouter(d_model),
    )

    h = torch.randn(B, T, d_model)
    ess = essence(B)
    out_h, metrics = stack(h, ess, return_step_metrics=True)

    assert "gate_entropy_loss" in metrics
    assert metrics["gate_entropy_loss"].requires_grad
    assert "_exit_entropy_tensor" in metrics
    assert metrics["_exit_entropy_tensor"].requires_grad


# 6) RoPE + SDPA
def check_rope_sdpa() -> None:
    from tajalli.model.tajalli_model import TajalliModelPhase1
    from tajalli.model.init import init_tajalli_weights

    model = TajalliModelPhase1(
        vocab_size=50257,
        d_model=64,
        d_essence=64,
        n_heads=4,
        d_head=16,
        d_ff=128,
        n_steps=4,
    )
    init_tajalli_weights(model, n_steps=4)

    attn = model.tajalli_stack.block.attention.attn
    assert attn.rope is not None
    assert attn.backend == "sdpa"


# 7) Scheduler lr_final (covered in test_scheduler.py; quick inline check)
def check_scheduler_lr_final() -> None:
    from tajalli.training.scheduler import get_cosine_warmup_scheduler

    base_lr = 6e-4
    lr_final = 1e-5
    min_ratio = lr_final / base_lr
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps=100, total_steps=1000, min_lr_ratio=min_ratio)
    for _ in range(999):
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    actual = optimizer.param_groups[0]["lr"]
    assert abs(actual - lr_final) < 1e-6, f"Final LR={actual}, expected {lr_final}"


def main() -> int:
    checks = [
        ("matrix essence + attribute heads", check_matrix_essence_and_heads),
        ("depth families", check_depth_families),
        ("alpha schedule", check_alpha_schedule),
        ("inner layers", check_inner_layers),
        ("gate/exit entropy wiring", check_gate_exit_entropy_wiring),
        ("RoPE + SDPA", check_rope_sdpa),
        ("scheduler lr_final", check_scheduler_lr_final),
    ]
    failed = []
    for name, fn in checks:
        try:
            fn()
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} check(s) failed: {failed}")
        return 1
    print(f"\nAll {len(checks)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
