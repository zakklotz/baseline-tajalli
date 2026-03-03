#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from tajalli.data.dataloader import (
    get_wikitext103_dataset_meta,
    get_wikitext103_dataloaders,
    get_wikitext103_tokenization_meta,
    get_wikitext103_tokenizer,
)
from tajalli.data.freq_vocab import FreqArtifact, load_freq_artifact, validate_freq_artifact
from tajalli.model.init import init_tajalli_weights
from tajalli.model.tajalli_model import TajalliModelPhase1
from tajalli.training.config_utils import (
    NormalizationReport,
    apply_lawh_every_k,
    normalize_phase1_config,
    print_normalization_report,
    print_training_config,
    validate_phase1_config,
)
from tajalli.training.trainer import Phase1Trainer


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML object at {path}, got {type(data).__name__}.")
    return data


def resolve_resume_path(config: dict[str, Any], cli_resume: str | None) -> str | None:
    """Prefer the CLI resume path; otherwise honor resume_from from config."""
    return cli_resume if cli_resume is not None else config.get("resume_from")


def preflight_phase1_config(
    config_path: str | Path,
    *,
    device_override: str | None = None,
) -> tuple[dict[str, Any], NormalizationReport, Any, FreqArtifact | None]:
    """Normalize and validate the Phase 1 config before constructing the trainer."""
    raw_config = load_yaml(config_path)
    config, report = normalize_phase1_config(raw_config, source_path=config_path)

    if device_override is not None:
        config["device"] = device_override

    apply_lawh_every_k(config)

    device = config.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config["device"] = device
        report.defaults_applied.append(f"device <- '{device}'")

    tokenizer = get_wikitext103_tokenizer(config["tokenizer_name"])
    vocab_size = int(len(tokenizer))
    configured_vocab_size = config.get("vocab_size")
    if configured_vocab_size is not None and int(configured_vocab_size) != vocab_size:
        raise ValueError(
            f"Config vocab_size ({configured_vocab_size}) does not match tokenizer "
            f"'{config['tokenizer_name']}' vocab size ({vocab_size})."
        )
    if configured_vocab_size is None:
        report.defaults_applied.append(f"vocab_size <- tokenizer ({vocab_size})")
    config["vocab_size"] = vocab_size

    if config.get("use_adaptive_softmax") and config.get("adaptive_softmax_cutoffs") is None:
        config["adaptive_softmax_cutoffs"] = [20000, 40000]
        report.defaults_applied.append("adaptive_softmax_cutoffs <- [20000, 40000]")

    validate_phase1_config(config)

    artifact = None
    if config.get("use_adaptive_softmax"):
        artifact = _validate_adaptive_softmax_preflight(config, tokenizer)

    return config, report, tokenizer, artifact


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

    cfg, report, _tokenizer, artifact = preflight_phase1_config(
        args.config,
        device_override=args.device,
    )

    seed = int(cfg.get("seed", 1337))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print_normalization_report(report)
    print_training_config(cfg, title="TAJALLI PHASE1 TRAIN CONFIG")

    cache_dir = cfg.get("cache_dir", "data/cache")
    tokenizer_name = cfg["tokenizer_name"]
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

    model = TajalliModelPhase1(
        vocab_size=int(cfg["vocab_size"]),
        d_model=int(cfg["d_model"]),
        d_essence=int(cfg["d_essence"]),
        n_heads=int(cfg["n_heads"]),
        d_head=int(cfg["d_head"]),
        d_ff=int(cfg["d_ff"]),
        n_steps=int(cfg["n_steps"]),
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
        use_adaptive_softmax=bool(cfg.get("use_adaptive_softmax", False)),
        freq_vocab_path=cfg.get("freq_vocab_path"),
        adaptive_softmax_cutoffs=cfg.get("adaptive_softmax_cutoffs"),
        adaptive_softmax_div_value=float(cfg.get("adaptive_softmax_div_value", 4.0)),
    )

    if cfg.get("use_adaptive_softmax"):
        if model.adaptive_output is None:
            raise RuntimeError(
                "use_adaptive_softmax=true but TajalliModelPhase1 did not construct adaptive_output."
            )
        _print_adaptive_softmax_banner(cfg, artifact)
    else:
        print("[adaptive-softmax] disabled")

    resume_path = resolve_resume_path(cfg, args.resume)

    if resume_path is None:
        init_tajalli_weights(
            model,
            std=float(cfg.get("init_std", 0.02)),
            n_steps=int(cfg["recursive_steps"]),
        )

    run_name = cfg.get("run_name", "tajalli_phase1")
    log_dir = Path(cfg.get("log_dir", "logs")) / run_name
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints")) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = Phase1Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        log_dir=str(log_dir),
        checkpoint_dir=str(ckpt_dir),
        model_name=run_name,
        essence_warmup_steps=cfg.get("essence_warmup_steps"),
    )

    start_step = maybe_resume(trainer, resume_path)
    max_steps = int(cfg["max_steps"])
    if start_step >= max_steps:
        print(f"[done] resume step {start_step} >= max_steps {max_steps}")
        return

    trainer.train()


def _validate_adaptive_softmax_preflight(config: dict[str, Any], tokenizer) -> FreqArtifact:
    freq_vocab_path = config.get("freq_vocab_path")
    if not freq_vocab_path:
        raise ValueError("use_adaptive_softmax=true requires freq_vocab_path.")

    path = Path(freq_vocab_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Adaptive-softmax freq artifact not found at {path}. "
            "Rebuild it with scripts/build_wt103_freq.py."
        )

    artifact = load_freq_artifact(path)
    dataset_meta = get_wikitext103_dataset_meta(split="train")
    validate_freq_artifact(
        artifact,
        vocab_size=int(config["vocab_size"]),
        tokenizer=tokenizer,
        tokenizer_name=config["tokenizer_name"],
        dataset_name=dataset_meta["name"],
        dataset_config=dataset_meta["config"],
        split=dataset_meta["split"],
        text_field=dataset_meta["text_field"],
        tokenization=get_wikitext103_tokenization_meta(tokenizer),
        strict_training=True,
    )
    return artifact


def _print_adaptive_softmax_banner(config: dict[str, Any], artifact: FreqArtifact | None) -> None:
    if artifact is None:
        raise RuntimeError("Adaptive softmax banner requires a validated frequency artifact.")

    dataset = artifact.dataset or {}
    print(
        "[adaptive-softmax] enabled "
        f"(schema=v{artifact.schema_version}, tokenizer={config['tokenizer_name']}, "
        f"vocab_size={config['vocab_size']}, dataset={dataset.get('name')}/{dataset.get('split')}, "
        f"cutoffs={config.get('adaptive_softmax_cutoffs')})"
    )


if __name__ == "__main__":
    main()
