import torch
from tajalli.model.tajalli_model import TajalliModelPhase1


def test_phase1_forward_smoke():
    torch.manual_seed(0)

    # Keep this tiny so it's CPU friendly
    model = TajalliModelPhase1(
        vocab_size=128,
        d_model=64,
        n_heads=8,
        d_head=8,
        d_essence=32,
        n_inner_layers=2,
        d_ff=256,
        dropout=0.0,
        max_seq_len=32,
    )

    x = torch.randint(0, 128, (2, 16))
    out = model(x)

    assert out is not None

    # TajalliModelPhase1 may return logits or (logits, aux)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out

    assert logits.shape[0] == 2
    assert logits.shape[-1] == 128
