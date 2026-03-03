"""Tests for recursive KV cache (reuse K,V across recursion steps)."""

import torch

from tajalli.model.tajalli import TajalliBlock
from tajalli.model.essence import EssenceCoreMatrix


def test_recursive_kv_cache_step0_returns_kv():
    """Step 0 with return_kv=True should return (h, metrics, (K,V))."""
    B, T, d_model, d_essence = 2, 8, 64, 64
    n_heads, d_head, d_ff = 4, 16, 128
    essence = EssenceCoreMatrix(n_rows=32, d_model=d_model)
    block = TajalliBlock(
        d_model=d_model,
        d_essence=d_model,
        n_heads=n_heads,
        d_head=d_head,
        d_ff=d_ff,
        essence_type="matrix",
        n_essence_rows=32,
        n_inner_layers=0,
    )
    h = torch.randn(B, T, d_model) * 0.1
    ess = essence(B)

    out = block(h, ess, return_metrics=False, step_idx=0, return_kv=True, memory_h=None)
    assert len(out) == 3
    h_new, _, kv = out
    k, v = kv
    assert k.shape == (B, n_heads, T, d_head)
    assert v.shape == (B, n_heads, T, d_head)


def test_recursive_kv_cache_step1_reuses_kv():
    """Step 1 with cached_kv from step 0 should run without recomputing K,V."""
    B, T, d_model, d_essence = 2, 8, 64, 64
    n_heads, d_head, d_ff = 4, 16, 128
    essence = EssenceCoreMatrix(n_rows=32, d_model=d_model)
    block = TajalliBlock(
        d_model=d_model,
        d_essence=d_model,
        n_heads=n_heads,
        d_head=d_head,
        d_ff=d_ff,
        essence_type="matrix",
        n_essence_rows=32,
        n_inner_layers=0,
    )
    h = torch.randn(B, T, d_model) * 0.1
    ess = essence(B)

    # Step 0: get kv
    _, _, kv = block(h, ess, return_metrics=False, step_idx=0, return_kv=True, memory_h=None)

    # Step 1: reuse kv
    out = block(h, ess, return_metrics=False, step_idx=1, cached_kv=kv, memory_h=None)
    assert len(out) == 2
    h_new, _ = out
    assert h_new.shape == (B, T, d_model)
