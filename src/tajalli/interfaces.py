from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

import torch


@runtime_checkable
class LanguageModel(Protocol):
    def __call__(self, input_ids: torch.Tensor, **kwargs: Any) -> Any: ...


def extract_logits(model_out: Any) -> torch.Tensor:
    """
    Normalize different model output formats into logits tensor (B, T, V).

    Supported:
      - torch.Tensor
      - {"logits": tensor} / {"lm_logits": tensor} / {"out": tensor}
      - objects with .logits
      - (logits, *rest)
    """
    if isinstance(model_out, torch.Tensor):
        return model_out

    if isinstance(model_out, tuple) and len(model_out) >= 1:
        first = model_out[0]
        if isinstance(first, torch.Tensor):
            return first
        if isinstance(first, dict):
            model_out = first  # fall through to dict case

    if isinstance(model_out, dict):
        d: Dict[str, Any] = model_out
        for key in ("logits", "lm_logits", "out"):
            if key in d and isinstance(d[key], torch.Tensor):
                return d[key]
        raise KeyError(f"Dict output missing logits-like key. Keys={list(d.keys())}")

    if hasattr(model_out, "logits") and isinstance(model_out.logits, torch.Tensor):
        return model_out.logits

    raise TypeError(f"Expected logits as torch.Tensor / dict / tuple / .logits, got {type(model_out)}")
