from __future__ import annotations

from pathlib import Path
from typing import Any


def find_parity_config_dir(baseline_root: Path) -> Path:
    p = baseline_root / "configs" / "parity"
    if not p.exists():
        raise FileNotFoundError(f"Could not find baseline parity configs at: {p}")
    return p


def choose_parity_config_file(parity_dir: Path) -> Path:
    """
    Pick a parity config file.

    Preference:
      - wt103_512d_standard.yaml if present (stable default)
      - otherwise the smallest yaml/yml (often the minimal one)
    """
    preferred = parity_dir / "wt103_512d_standard.yaml"
    if preferred.exists():
        return preferred

    yamls = sorted(list(parity_dir.glob("*.yaml")) + list(parity_dir.glob("*.yml")))
    if not yamls:
        raise FileNotFoundError(f"No .yaml/.yml files found in: {parity_dir}")
    return sorted(yamls, key=lambda x: x.stat().st_size)[0]


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # requires PyYAML

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict at root of yaml: {path}")
    return data


def extract_model_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Baseline configs may use:
      - top-level keys
      - model: {...}
      - cfg: {...}
    """
    model_dict = raw.get("model", raw.get("cfg", raw))
    if not isinstance(model_dict, dict):
        raise TypeError(f"Expected dict model cfg, got {type(model_dict)}")
    return dict(model_dict)


def coerce_cfg_dict(cfg_dict: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Apply overrides only for keys already present in cfg_dict.
    """
    out = dict(cfg_dict)
    for k, v in overrides.items():
        if k in out:
            out[k] = v
    return out
