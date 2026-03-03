from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def save_resolved_config(config: dict[str, Any], run_dir: Path) -> None:
    path = run_dir / "config.resolved.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_command(run_dir: Path) -> None:
    path = run_dir / "command.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write(shlex.join(sys.argv) + "\n")


def save_git_commit(run_dir: Path) -> None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return

    commit = out.stdout.strip()
    if not commit:
        return

    path = run_dir / "git_commit.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write(commit + "\n")


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def save_model_stats(
    run_dir: Path,
    *,
    model_type: str,
    recursive: bool,
    recurrence_steps: int,
    total_params: int,
    trainable_params: int,
    effective_batch_size: int,
) -> str:
    lines = [
        f"model_type: {model_type}",
        f"recursive: {recursive}",
        f"recurrence_steps: {recurrence_steps}",
        f"total_params: {total_params}",
        f"trainable_params: {trainable_params}",
        f"effective_batch_size: {effective_batch_size}",
    ]
    text = "\n".join(lines) + "\n"
    path = run_dir / "model_stats.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    return text
