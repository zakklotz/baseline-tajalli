#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from tajalli.data.dataloader import get_wikitext103_dataloaders
from tajalli.model.tajalli_model import TajalliModelPhase1
from tajalli.model.init import init_tajalli_weights
from tajalli.training.trainer import Phase1Trainer
from tajalli.training.config_utils import apply_lawh_every_k, print_training_config


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def maybe_resume(trainer: Phase1Trainer, resume_path: str | None) -> int:
    if not resume_path:
        return 0
    ckpt = torch.load(resume_path, map_location="cpu")
    trainer.model.load_state_dict(ckpt["model"])
    trainer.optimizer.load_state_dict(ckpt["optimizer"])
    trainer.scheduler.load_state_dict(ckpt["scheduler"])
    step = int(ckpt.get("step", 0))
    print(f"[resume] loaded checkpoint {resume_path} at step={step}")
    return step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config path")
    ap.add_argument("--device", type=str, default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # Optional CLI override
    if args.device is not None:
        cfg["device"] = args.device

    # Optional: convert lawh_every_k_steps -> lawh_at_steps
    apply_lawh_every_k(cfg)

    # Device default
    device = cfg.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg["device"] = device

    # Seed (your dataloader doesn't take it, so set torch seed here)
    seed = int(cfg.get("seed", 1337))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print_training_config(cfg, title="TAJALLI PHASE1 TRAIN CONFIG")

    # Data (match tajalli.data.dataloader signature exactly)
    cache_dir = cfg.get("cache_dir", "data/cache")
    tokenizer_name = cfg.get("tokenizer_name", "gpt2")
    max_seq_len = int(cfg.get("seq_len", cfg.get("max_seq_len", 512)))

    train_loader, val_loader = get_wikitext103_dataloaders(
        tokenizer_name=tokenizer_name,
        max_seq_len=max_seq_len,
        batch_size=int(cfg["batch_size"]),
        num_workers=cfg.get("num_workers", None),
        pin_memory=cfg.get("pin_memory", None),
        show_progress=bool(cfg.get("show_progress", True)),
        cache_dir=cache_dir,
    )

    # Determine vocab size from tokenizer if not provided
    if "vocab_size" in cfg and cfg["vocab_size"] is not None:
        vocab_size = int(cfg["vocab_size"])
    else:
        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        vocab_size = int(len(tok))

    # Model
    model = TajalliModelPhase1(
        vocab_size=vocab_size,
        d_model=int(cfg["d_model"]),
        d_essence=int(cfg["d_essence"]),
        n_heads=int(cfg["n_heads"]),
        d_head=int(cfg["d_head"]),
        d_ff=int(cfg["d_ff"]),
        n_steps=int(cfg.get("n_steps", 6)),
        max_seq_len=int(cfg.get("max_seq_len", max_seq_len)),
        dropout=float(cfg.get("dropout", 0.0)),
        padding_idx=cfg.get("padding_idx", None),
        essence_init=str(cfg.get("essence_init", "spectral")),
        essence_path=cfg.get("essence_path", None),
        essence_type=cfg.get("essence_type", "vector"),
        n_essence_rows=int(cfg.get("n_essence_rows", 64)),
        alpha_schedule=cfg.get("alpha_schedule", None),
        n_inner_layers=int(cfg.get("n_inner_layers", 0)),
        use_exit_router=bool(cfg.get("use_exit_router", False)),
        exit_threshold=float(cfg.get("exit_threshold", 0.5)),
        exit_capacity_fraction=float(cfg.get("exit_capacity_fraction", 0.5)),
        use_recursive_kv_cache=bool(cfg.get("use_recursive_kv_cache", False)),
        depth_families=cfg.get("depth_families", None),
        family_steps=cfg.get("family_steps", None),
        hypernetwork_attributes=bool(cfg.get("hypernetwork_attributes", False)),
    )

    # Only init when starting fresh (never when resuming)
    if args.resume is None:
        # Use recursive_steps if present, else n_steps
        n_steps = int(cfg.get("recursive_steps", cfg.get("n_steps", 6)))
        init_tajalli_weights(
            model,
            std=float(cfg.get("init_std", 0.02)),
            n_steps=n_steps,
        )

    # Paths
    run_name = cfg.get("run_name", "tajalli_phase1")
    log_dir = Path(cfg.get("log_dir", "logs")) / run_name
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints")) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Trainer
    trainer = Phase1Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        log_dir=str(log_dir),
        checkpoint_dir=str(ckpt_dir),
        model_name=run_name,
    )

    start_step = maybe_resume(trainer, args.resume)
    max_steps = int(cfg["max_steps"])
    if start_step >= max_steps:
        print(f"[done] resume step {start_step} >= max_steps {max_steps}")
        return

    trainer.train()


if __name__ == "__main__":
    main()
