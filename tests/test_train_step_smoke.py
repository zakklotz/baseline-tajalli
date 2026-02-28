import torch
import torch.nn.functional as F

from tajalli.model.tajalli_model import TajalliModelPhase1


def test_single_train_step_smoke_cpu():
    torch.manual_seed(0)

    vocab_size = 128
    model = TajalliModelPhase1(
        vocab_size=vocab_size,
        d_model=64,
        d_essence=32,
        n_heads=8,
        d_head=8,
        d_ff=256,
        n_steps=2,         # keep tiny
        max_seq_len=32,
        dropout=0.0,
        n_inner_layers=1,  # small but non-trivial
    )
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Fake batch: next-token prediction
    B, T = 2, 16
    x = torch.randint(0, vocab_size, (B, T))
    y = torch.randint(0, vocab_size, (B, T))

    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out

    assert logits.shape[:2] == (B, T)
    assert logits.shape[-1] == vocab_size

    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    assert torch.isfinite(loss)

    optim.zero_grad(set_to_none=True)
    loss.backward()

    # sanity: at least one grad exists and is finite
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)

    optim.step()
