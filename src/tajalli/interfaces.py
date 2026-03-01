from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class LanguageModel(Protocol):
    def __call__(self, input_ids: torch.Tensor, **kwargs: Any) -> Any: ...


def extract_logits(model_out: Any) -> torch.Tensor:
    """
    Normalize different model output conventions into logits tensor.

    Supported:
      - torch.Tensor -> logits
      - (logits, ...) tuple/list -> logits is [0]
      - {"logits": logits, ...} dict -> logits
      - {"y": logits, ...} dict -> logits (common alt)
    """
    if isinstance(model_out, torch.Tensor):
        return model_out

    if isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        first = model_out[0]
        if isinstance(first, torch.Tensor):
            return first

    if isinstance(model_out, dict):
        for key in ("logits", "y", "output", "out"):
            if key in model_out and isinstance(model_out[key], torch.Tensor):
                return model_out[key]
        # Sometimes nested: {"model": {"logits": ...}}
        if "model" in model_out and isinstance(model_out["model"], dict):
            inner = model_out["model"]
            if "logits" in inner and isinstance(inner["logits"], torch.Tensor):
                return inner["logits"]

        raise TypeError(
            f"Dict output did not contain a tensor logits under keys "
            f"('logits','y','output','out'). Keys={list(model_out.keys())}"
        )

    raise TypeError(f"Expected logits as torch.Tensor/tuple/list/dict, got {type(model_out)}")
