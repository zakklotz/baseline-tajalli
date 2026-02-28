import torch
from tajalli.nncore_attention import TajalliAttention


def test_attention_shape():
    B, T, C = 2, 8, 64
    attn = TajalliAttention(d_model=C, n_heads=8, dropout=0.0)

    x = torch.randn(B, T, C)
    y = attn(x)

    assert y.shape == (B, T, C)


def test_attention_deterministic():
    torch.manual_seed(0)

    B, T, C = 2, 8, 64
    attn = TajalliAttention(d_model=C, n_heads=8, dropout=0.0)

    x = torch.randn(B, T, C)

    y1 = attn(x)
    y2 = attn(x)

    assert torch.allclose(y1, y2)
