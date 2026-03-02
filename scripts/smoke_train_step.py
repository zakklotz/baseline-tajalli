from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F

from tajalli.interfaces import extract_logits
from scripts._baseline_config import load_experiment_config
from scripts._baseline_imports import add_baseline_transformer_to_syspath


@dataclass
class SmokeCfg:
    vocab_size: int = 128
    seq_len: int = 64
    batch_size: int = 2
    d_model: int = 64
    n_heads: int = 8
    d_head: int = 8
    d_ff: int = 256
    d_essence: int = 32
    n_steps: int = 2
    max_seq_len: int = 64
    dropout: float = 0.0
    n_inner_layers: int = 0


def build_tajalli(cfg: SmokeCfg) -> torch.nn.Module:
    from tajalli.model.tajalli_model import TajalliModelPhase1

    return TajalliModelPhase1(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        d_essence=cfg.d_essence,
        n_heads=cfg.n_heads,
        d_head=cfg.d_head,
        d_ff=cfg.d_ff,
        n_steps=cfg.n_steps,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        n_inner_layers=cfg.n_inner_layers,
    )


def build_baseline(cfg, repo_root: Path) -> torch.nn.Module:
    """Build the *baseline-transformer* model from its resolved run config."""
    add_baseline_transformer_to_syspath(repo_root)

    from baseline_transformer.config import ExperimentConfig

    if not isinstance(cfg, ExperimentConfig):
        raise TypeError(f"Expected ExperimentConfig, got {type(cfg)}")

    # Prefer baseline_transformer.train.build helpers if present.
    build_mod = None
    try:
        from baseline_transformer.train import build as build_mod  # type: ignore
    except Exception:
        build_mod = None

    if build_mod is not None:
        for fn_name in ("build_model", "build"):
            fn = getattr(build_mod, fn_name, None)
            if fn is None:
                continue
            try:
                sig = inspect.signature(fn)
                if len(sig.parameters) == 1:
                    model = fn(cfg)
                else:
                    kwargs = {}
                    if "cfg" in sig.parameters:
                        kwargs["cfg"] = cfg
                    if "exp" in sig.parameters:
                        kwargs["exp"] = cfg
                    model = fn(**kwargs) if kwargs else fn(cfg)

                if isinstance(model, torch.nn.Module):
                    return model
            except Exception:
                pass

    # Fallback: instantiate directly
    from baseline_transformer.models.standard import StandardTransformerLM
    from baseline_transformer.models.recursive import RecursiveTransformerLM

    arch = str(cfg.model.get("arch", "standard")).lower()
    if arch in ("recursive", "rec", "r"):
        return RecursiveTransformerLM(cfg)  # type: ignore[arg-type]
    return StandardTransformerLM(cfg)  # type: ignore[arg-type]


def one_step(model: torch.nn.Module, vocab_size: int, device: torch.device) -> float:
    model.train()
    x = torch.randint(0, vocab_size, (2, 16), device=device)
    y = torch.randint(0, vocab_size, (2, 16), device=device)

    out = model(x)
    logits = extract_logits(out)

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    return float(loss.item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--model", default="tajalli", choices=["tajalli", "baseline"])
    args = p.parse_args()

    device = torch.device(args.device)

    repo_root = Path(__file__).resolve().parents[1]

    if args.model == "tajalli":
        cfg = SmokeCfg()
        model = build_tajalli(cfg).to(device)
        vocab_size = cfg.vocab_size
    else:
        # Use a resolved baseline run config (keeps parity)
        resolved = Path("../baseline-transformer/runs/wt103_512d_standard/config.resolved.yaml").resolve()
        print(f"[baseline] using resolved run config: {resolved}")
        cfg = load_experiment_config(resolved, repo_root=repo_root)
        model = build_baseline(cfg, repo_root=repo_root).to(device)
        vocab_size = int(cfg.model.get("vocab_size", 50257))

    for step in range(args.steps):
        loss = one_step(model, vocab_size=vocab_size, device=device)
        print(f"step={step} loss={loss:.4f}")

    print("ok")


if __name__ == "__main__":
    main()
