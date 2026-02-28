#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F

from tajalli.model.tajalli_model import TajalliModelPhase1


def main():
    ap = argparse.ArgumentParser(description="Tajalli smoke train-step (CPU/GPU)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--vocab-size", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    torch.manual_seed(0)

    model = TajalliModelPhase1(
        vocab_size=args.vocab_size,
        d_model=64,
        d_essence=32,
        n_heads=8,
        d_head=8,
        d_ff=256,
        n_steps=2,
        max_seq_len=32,
        dropout=0.0,
        n_inner_layers=1,
    ).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for i in range(args.steps):
        x = torch.randint(0, args.vocab_size, (args.batch, args.seq_len), device=device)
        y = torch.randint(0, args.vocab_size, (args.batch, args.seq_len), device=device)

        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out

        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        print(f"step={i} loss={loss.item():.4f}")

    print("ok")


if __name__ == "__main__":
    main()
