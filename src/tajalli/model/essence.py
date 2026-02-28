"""Frozen Essence Core for Tajallī architecture."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Literal, Optional


def _spectral_init(shape: tuple, device: torch.device) -> torch.Tensor:
    """Random init with SVD and 1/k singular value decay."""
    d = shape[-1]
    x = torch.randn(*shape, device=device)
    if len(shape) == 1:
        x = x.unsqueeze(0)
    u, s, vh = torch.linalg.svd(x.float(), full_matrices=False)
    # 1/k decay for singular values
    k = torch.arange(1, s.shape[-1] + 1, device=device, dtype=s.dtype)
    s_scaled = s / k
    result = (u * s_scaled.unsqueeze(-2)) @ vh
    if len(shape) == 1:
        result = result.squeeze(0)
    return result


class EssenceCore(nn.Module):
    """
    Frozen essence parameter. Never updated by optimizer.
    Supports batch expansion via .expand(B, ...).
    """

    def __init__(
        self,
        d_essence: int,
        n_slots: int = 1,
        init: Literal["spectral", "distilled_qwen", "distilled_r1", "svd_hidden", "learned_then_frozen"] = "spectral",
        path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.d_essence = d_essence
        self.n_slots = n_slots
        self.init_strategy = init

        if init == "spectral":
            data = _spectral_init((n_slots, d_essence), device or torch.device("cpu"))
        elif init in ("distilled_qwen", "distilled_r1", "svd_hidden"):
            if path is None:
                raise ValueError(
                    f"Must provide path to pre-computed essence for init={init}. "
                    "Run scripts/init_essence_from_hidden_states.py or scripts/distill_essence.py first."
                )
            if init == "svd_hidden":
                data = self._load_svd_hidden(path, device)
            else:
                data = self._load_distilled(path, device)
        elif init == "learned_then_frozen":
            # Start with spectral, will be trained then frozen
            data = _spectral_init((n_slots, d_essence), device or torch.device("cpu"))
        else:
            raise ValueError(f"Unknown init: {init}")

        requires_grad = init == "learned_then_frozen"
        self.essence = nn.Parameter(data, requires_grad=requires_grad)

    def _load_svd_hidden(self, path: str, device: Optional[torch.device]) -> torch.Tensor:
        """Load essence from init_essence_from_hidden_states.py output (shape n_slots x d_essence)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"SVD essence not found at {path}. Run: python scripts/init_essence_from_hidden_states.py"
            )
        data = torch.load(p, map_location=device or "cpu", weights_only=True)
        if isinstance(data, dict) and "essence" in data:
            data = data["essence"]
        tensor = data if isinstance(data, torch.Tensor) else torch.tensor(data)
        if tensor.shape[1] != self.d_essence:
            raise ValueError(
                f"SVD essence d_essence {tensor.shape[1]} != expected {self.d_essence}"
            )
        if tensor.shape[0] != self.n_slots:
            if tensor.shape[0] == 1 and self.n_slots == 1:
                pass
            elif tensor.shape[0] > 1 and self.n_slots == 1:
                tensor = tensor.mean(dim=0, keepdim=True)
            else:
                tensor = tensor[: self.n_slots]
        return tensor

    def _load_distilled(self, path: str, device: Optional[torch.device]) -> torch.Tensor:
        """Load essence from distill_essence.py output."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Distilled essence not found at {path}. Run: python scripts/distill_essence.py"
            )
        data = torch.load(p, map_location=device or "cpu", weights_only=True)
        if isinstance(data, dict) and "essence" in data:
            data = data["essence"]
        tensor = data if isinstance(data, torch.Tensor) else torch.tensor(data)
        if tensor.shape != (self.n_slots, self.d_essence):
            # Allow (d_essence,) or (n_slots, d_essence)
            if tensor.shape == (self.d_essence,):
                tensor = tensor.unsqueeze(0).expand(self.n_slots, -1)
            else:
                raise ValueError(
                    f"Loaded essence shape {tensor.shape} != expected ({self.n_slots}, {self.d_essence})"
                )
        return tensor

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return (B, n_slots, d_essence) or (B, d_essence) if n_slots==1."""
        e = self.essence
        if batch_size > 1:
            e = e.unsqueeze(0).expand(batch_size, -1, -1)
        if self.n_slots == 1:
            e = e.squeeze(1)  # (B, d_essence)
        return e

    def expand_for_batch(self, batch_size: int) -> torch.Tensor:
        """Alias for forward for API clarity."""
        return self.forward(batch_size)

    def freeze(self) -> None:
        """Freeze the essence (for learned_then_frozen after warmup)."""
        self.essence.requires_grad = False


class EssenceCoreMatrix(nn.Module):
    """
    Frozen matrix essence: orthogonal rows in d_model space.
    Shape (n_rows, d_model). Never updated by optimizer.
    """

    def __init__(self, n_rows: int = 64, d_model: int = 512, row_norm_scale: float = 32.4):
        super().__init__()
        self.n_rows = n_rows
        self.d_model = d_model
        raw = torch.randn(n_rows, d_model)
        Q, _ = torch.linalg.qr(raw.T)  # [d_model, n_rows]
        essence = Q.T  # [n_rows, d_model] — orthogonal rows
        scale = row_norm_scale / (n_rows ** 0.5)
        essence = essence * scale
        self.essence = nn.Parameter(essence, requires_grad=False)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return (B, n_rows, d_model)."""
        return self.essence.unsqueeze(0).expand(batch_size, -1, -1)

    def expand_for_batch(self, batch_size: int) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(batch_size)
