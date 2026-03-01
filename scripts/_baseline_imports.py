from __future__ import annotations

import os
import sys
from pathlib import Path


def locate_baseline_root(repo_root: Path) -> Path:
    """
    Locate the baseline-transformer repo root.

    Supported layouts:
      - legacy: <repo_root>/.baseline-transformer
      - sibling: <repo_root>/../baseline-transformer

    Optional override:
      - BASELINE_TRANSFORMER_ROOT=/abs/path/to/baseline-transformer
    """
    env = os.environ.get("BASELINE_TRANSFORMER_ROOT")
    if env:
        p = Path(env)
        if (p / "src").exists() and (p / "configs").exists():
            return p.resolve()

    cands = [
        repo_root / ".baseline-transformer",
        repo_root.parent / "baseline-transformer",
    ]
    for p in cands:
        if (p / "src").exists() and (p / "configs").exists():
            return p.resolve()

    tried = "\n  - " + "\n  - ".join(str(c) for c in cands)
    raise FileNotFoundError(
        "Could not locate baseline-transformer repo root.\n"
        f"Tried:{tried}\n\n"
        "Fix options:\n"
        "  1) Place baseline-transformer at ../baseline-transformer\n"
        "  2) Or set BASELINE_TRANSFORMER_ROOT=/path/to/baseline-transformer\n"
    )


def add_baseline_transformer_to_syspath(repo_root: Path) -> Path:
    """
    Add baseline-transformer/src to sys.path and return the resolved src path.

    Optional override:
      - BASELINE_TRANSFORMER_SRC=/abs/path/to/baseline-transformer/src
    """
    env = os.environ.get("BASELINE_TRANSFORMER_SRC")
    if env:
        p = Path(env)
        if p.exists():
            rp = str(p.resolve())
            if rp not in sys.path:
                sys.path.insert(0, rp)
            return p.resolve()

    baseline_root = locate_baseline_root(repo_root)
    baseline_src = baseline_root / "src"
    rp = str(baseline_src.resolve())
    if rp not in sys.path:
        sys.path.insert(0, rp)
    return baseline_src.resolve()
