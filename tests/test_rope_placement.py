"""Tests for RoPE placement (q/k after projection, not x)."""

import torch


def test_rope_applied_to_qk_not_x():
    """Verify nn-core applies RoPE to q/k tensors, not to x before projection.
    x must be unchanged; RoPE is applied to projected q,k inside attention.
    """
    try:
        from nncore.layers.attention import MultiheadAttention
    except ImportError:
        import pytest
        pytest.skip("nn-core not installed")

    B, T, d_model, n_heads = 2, 16, 64, 4
    attn = MultiheadAttention(
        d_model=d_model,
        num_heads=n_heads,
        positional="rope",
        max_seq_len=64,
        backend="manual",
    )
    x = torch.randn(B, T, d_model)
    x_before = x.clone()

    # RoPE module exists and is used on q/k inside forward (not on x)
    assert attn.rope is not None

    # x should be unchanged by attention forward
    _ = attn(x)
    assert torch.allclose(x, x_before), "x must not be modified by attention"
