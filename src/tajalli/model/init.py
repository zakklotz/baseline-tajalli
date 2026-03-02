"""
Initialization utilities for Tajalli models.

Why this exists:
- A plain random LM should start near log(vocab) ~ 10-11 for GPT2 vocab.
- If you see loss >> 50 at step 0, your activations/logits are exploding.
- Recursive stacks can amplify variance. We damp residual-producing projections
  by 1/sqrt(n_steps) to keep early logits sane.

This init is intentionally "GPT-2-like":
- Linear/Embedding: N(0, 0.02)
- LayerNorm: weight=1, bias=0
- Biases: 0
- Residual output projections (attn.out_proj, FFN.out_proj / second linear): scaled down
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


def _iter_named_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    for name, mod in model.named_modules():
        yield name, mod


def _is_layernorm(mod: nn.Module) -> bool:
    return isinstance(mod, (nn.LayerNorm,))


def _init_layernorm(mod: nn.Module) -> None:
    # LayerNorm: weight=1, bias=0
    if hasattr(mod, "weight") and mod.weight is not None:
        nn.init.ones_(mod.weight)
    if hasattr(mod, "bias") and mod.bias is not None:
        nn.init.zeros_(mod.bias)


def _init_linear(mod: nn.Linear, std: float) -> None:
    nn.init.normal_(mod.weight, mean=0.0, std=std)
    if mod.bias is not None:
        nn.init.zeros_(mod.bias)


def _init_embedding(mod: nn.Embedding, std: float) -> None:
    nn.init.normal_(mod.weight, mean=0.0, std=std)
    if mod.padding_idx is not None:
        # Zero the pad row to avoid injecting signal
        with torch.no_grad():
            mod.weight[mod.padding_idx].zero_()


def _scale_weight(mod: nn.Module, attr: str, scale: float) -> None:
    w = getattr(mod, attr, None)
    if isinstance(w, torch.Tensor):
        with torch.no_grad():
            w.mul_(scale)


def init_tajalli_weights(
    model: nn.Module,
    *,
    std: float = 0.02,
    n_steps: int = 6,
) -> None:
    """
    Apply stable initialization across Tajalli + nn-core submodules.

    Key behavior:
    - Standard GPT-2-ish init everywhere.
    - Scale residual output projections by 1/sqrt(n_steps) to prevent recursion blow-up.
    """
    if n_steps <= 0:
        n_steps = 1

    # Damping factor for residual-producing projections.
    # Recursion behaves like stacking; this keeps early activations from exploding.
    damp = 1.0 / math.sqrt(float(n_steps))

    # 1) Generic init for common modules
    for name, mod in _iter_named_modules(model):
        if isinstance(mod, nn.Embedding):
            _init_embedding(mod, std=std)

        elif isinstance(mod, nn.Linear):
            _init_linear(mod, std=std)

        elif _is_layernorm(mod):
            _init_layernorm(mod)

    # 2) Targeted scaling for "residual output" projections
    # nn-core MultiheadAttention: out_proj
    for _, mod in _iter_named_modules(model):
        # nn-core attention has out_proj: nn.Linear
        if hasattr(mod, "out_proj") and isinstance(getattr(mod, "out_proj"), nn.Linear):
            # Only scale the *output* projection (residual path amplifier)
            _scale_weight(mod.out_proj, "weight", damp)

    # 3) Targeted scaling for FFN second projection if present
    # nn-core FFN/MLP variants often have something like "out_proj" or "fc2" or "w2"
    for _, mod in _iter_named_modules(model):
        # Common names
        if hasattr(mod, "fc2") and isinstance(getattr(mod, "fc2"), nn.Linear):
            _scale_weight(mod.fc2, "weight", damp)
        if hasattr(mod, "w2") and isinstance(getattr(mod, "w2"), nn.Linear):
            _scale_weight(mod.w2, "weight", damp)
        if hasattr(mod, "out_proj") and isinstance(getattr(mod, "out_proj"), nn.Linear):
            # Some FFNs also call the second projection out_proj; scaling twice is bad.
            # So only scale it here if it's not an attention module. Heuristic: attention has q_proj/k_proj/v_proj too.
            has_qkv = hasattr(mod, "q_proj") and hasattr(mod, "k_proj") and hasattr(mod, "v_proj")
            if not has_qkv:
                _scale_weight(mod.out_proj, "weight", damp)

    # 4) Tajalli-specific attribute heads often named row_weights/out_proj
    for _, mod in _iter_named_modules(model):
        if hasattr(mod, "row_weights") and isinstance(getattr(mod, "row_weights"), nn.Linear):
            _scale_weight(mod.row_weights, "weight", damp)
        if hasattr(mod, "out_proj") and isinstance(getattr(mod, "out_proj"), nn.Linear):
            # If it's an AttributeHeadSimple, scaling helps keep tajalli_signal sane.
            has_row = hasattr(mod, "row_weights")
            has_qkv = hasattr(mod, "q_proj") and hasattr(mod, "k_proj") and hasattr(mod, "v_proj")
            if has_row and not has_qkv:
                _scale_weight(mod.out_proj, "weight", damp)
