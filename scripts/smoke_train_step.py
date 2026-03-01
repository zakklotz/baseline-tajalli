#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from tajalli.interfaces import LanguageModel, extract_logits

from pathlib import Path
import sys
from pathlib import Path

# Ensure repo root is importable so we can import scripts helpers when running as a file.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._baseline_imports import add_baseline_transformer_to_syspath

@dataclass(frozen=True)
class SmokeCfg:
    vocab_size: int = 128
    d_model: int = 64
    d_essence: int = 32
    n_heads: int = 8
    d_head: int = 8
    d_ff: int = 256
    n_steps: int = 2
    max_seq_len: int = 32
    dropout: float = 0.0
    n_inner_layers: int = 1


def build_tajalli(cfg: SmokeCfg) -> LanguageModel:
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


def build_baseline(cfg: SmokeCfg, repo_root: Path) -> LanguageModel:
    from scripts._baseline_imports import add_baseline_transformer_to_syspath, locate_baseline_root

    # Make baseline package importable
    add_baseline_transformer_to_syspath(repo_root)
    baseline_root = locate_baseline_root(repo_root)

    # Use baseline’s resolved run config (stable + already in your runs/)
    cfg_path = baseline_root / "runs" / "wt103_512d_standard" / "config.resolved.yaml"
    print(f"[baseline] using resolved run config: {cfg_path}")

    # === Baseline’s own, real wiring ===
    from baseline_transformer.config import ExperimentConfig
    from baseline_transformer.nncore_bridge import build_transformer_config
    from baseline_transformer.models.standard import StandardTransformerLM

    exp = ExperimentConfig.load(cfg_path)

    # exp.model is the project-friendly dict; bridge converts it into nn-core TransformerConfig
    tcfg = build_transformer_config(exp.model)

    # Optional: enforce smoke overrides if you want parity at tiny sizes.
    # NOTE: Only do this if those fields exist on nn-core TransformerConfig.
    # Otherwise delete this block.
    if hasattr(tcfg, "vocab_size"):
        tcfg.vocab_size = cfg.vocab_size
    if hasattr(tcfg, "max_seq_len"):
        tcfg.max_seq_len = cfg.max_seq_len

    return StandardTransformerLM(tcfg)


def dynamic_import(spec: str):
    """
    Import "module.submodule:Symbol" and return the symbol.
    """
    if ":" not in spec:
        raise ValueError('Expected --baseline-import in form "module.path:SymbolName"')
    mod_name, sym_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, sym_name)


def main():
    ap = argparse.ArgumentParser(description="Tajalli smoke train-step (CPU/GPU)")

    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--model", default="tajalli", choices=["tajalli", "baseline"])

    # Optional dynamic baseline loader
    ap.add_argument(
        "--baseline-import",
        type=str,
        default=None,
        help='Baseline loader "module.path:ClassName" (if baseline is installable/importable).',
    )

    ap.add_argument("--vocab-size", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)

    args = ap.parse_args()

    repo_root = Path(__file__).parent.parent

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    torch.manual_seed(0)

    cfg = SmokeCfg(vocab_size=args.vocab_size)

    if args.model == "tajalli":
        model: LanguageModel = build_tajalli(cfg)
    else:
        if args.baseline_import is not None:
            BaselineCls = dynamic_import(args.baseline_import)
            model = BaselineCls(
                vocab_size=cfg.vocab_size,
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_ff=cfg.d_ff,
                max_seq_len=cfg.max_seq_len,
                dropout=cfg.dropout,
            )
        else:
            model = build_baseline(cfg, repo_root)

    model = model.to(device)  # type: ignore[attr-defined]
    model.train()  # type: ignore[attr-defined]

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)  # type: ignore[attr-defined]

    for i in range(args.steps):
        x = torch.randint(0, args.vocab_size, (args.batch, args.seq_len), device=device)
        y = torch.randint(0, args.vocab_size, (args.batch, args.seq_len), device=device)

        out = model(x)
        logits = extract_logits(out)

        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        print(f"step={i} loss={loss.item():.4f}")

    print("ok")


if __name__ == "__main__":
    main()
