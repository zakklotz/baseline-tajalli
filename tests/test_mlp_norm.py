import torch
from tajalli.nncore_mlp import TajalliFFN, build_norm


def test_ffn_shape():
    B, T, C = 2, 8, 64
    ffn = TajalliFFN(d_model=C, d_ff=256, dropout=0.0)
    x = torch.randn(B, T, C)
    y = ffn(x)
    assert y.shape == (B, T, C)


def test_norm_shape_and_finite():
    B, T, C = 2, 8, 64
    x = torch.randn(B, T, C)

    ln = build_norm("layernorm", C)
    y = ln(x)
    assert y.shape == (B, T, C)
    assert torch.isfinite(y).all()

    rn = build_norm("rmsnorm", C)
    z = rn(x)
    assert z.shape == (B, T, C)
    assert torch.isfinite(z).all()
