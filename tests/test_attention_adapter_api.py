import torch
from tajalli.nncore_attention import TajalliAttention


def test_attention_adapter_returns_tensor_when_no_kv():
    torch.manual_seed(0)
    attn = TajalliAttention(d_model=64, n_heads=8, dropout=0.0)

    x = torch.randn(2, 8, 64)
    y = attn(x, memory_h=None, cached_kv=None, return_kv=False)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, 8, 64)


def test_attention_adapter_returns_tuple_when_kv_requested():
    torch.manual_seed(0)
    attn = TajalliAttention(d_model=64, n_heads=8, dropout=0.0)

    x = torch.randn(2, 8, 64)
    y = attn(x, memory_h=None, cached_kv=None, return_kv=True)

    assert isinstance(y, tuple)
    assert len(y) == 2
    out, kv = y
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 8, 64)
    # nn-core returns (k, v) when return_kv=True for recursive KV cache
    assert kv is not None
    assert isinstance(kv, tuple)
    assert len(kv) == 2
    k, v = kv
    assert k.shape == (2, 8, 8, 8)
    assert v.shape == (2, 8, 8, 8)
