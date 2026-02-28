"""Lawḥ external memory: frozen key-value store + trainable cross-attention.

Two-stage retrieval: precompute top-K passage indices (e.g. via FAISS) per batch,
then cross-attend only over those K vectors. Lawḥ runs only at first/last recursive step.
"""

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tajalli.nncore_mlp import build_norm


class LawhMemoryStore(nn.Module):
    """
    Holds frozen key/value tensors. No trainable parameters.
    Load from file; get_keys_values(indices) returns (K, d_key), (K, d_value).
    Optional FAISS index for fast top-K retrieval by query-key similarity.
    """

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert keys.shape == values.shape, "keys and values must have same shape"
        self.register_buffer("_keys", keys)
        self.register_buffer("_values", values)
        self._d_key = keys.shape[1]
        self._d_value = values.shape[1]
        self._faiss_index = None  # Optional; set by load_from_file if present
        if device is not None:
            self._keys = self._keys.to(device)
            self._values = self._values.to(device)

    @property
    def d_key(self) -> int:
        return self._d_key

    @property
    def d_value(self) -> int:
        return self._d_value

    @property
    def num_passages(self) -> int:
        return self._keys.shape[0]

    def get_keys_values(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (K_keys, K_values) for the given passage indices.
        indices: (K,) or (B, K) long tensor
        Returns: (K, d_key), (K, d_value) on same device as store.
        """
        flat = indices.view(-1)
        K = self._keys[flat]
        V = self._values[flat]
        return K, V

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        k: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return top-K passage indices for each query. Uses FAISS if available, else brute-force.
        queries: (N, d_key) or (B, T, d_key) — will be flattened to (N, d_key)
        Returns: (N, k) long indices, or (N,) if k==1.
        """
        dev = device or queries.device
        q = queries.float().view(-1, self._d_key).to(dev)
        keys = self._keys.to(dev)
        # Normalize for cosine similarity = inner product
        q_n = F.normalize(q, p=2, dim=-1)
        k_n = F.normalize(keys, p=2, dim=-1)
        sim = q_n @ k_n.t()  # (N, num_passages)
        topk_sim, topk_idx = sim.topk(k, dim=-1)
        return topk_idx if k > 1 else topk_idx.squeeze(-1)

    @classmethod
    def load_from_file(
        cls,
        path: str,
        device: Optional[torch.device] = None,
        map_location: Optional[str] = None,
    ) -> "LawhMemoryStore":
        """Load keys/values (and optional FAISS index) from .pt file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Lawḥ store not found: {path}")
        # Lawh store is produced by our build_lawh_store.py; may contain numpy-backed
        # tensors, so we allow weights_only=False for this trusted file.
        data = torch.load(p, map_location=map_location or "cpu", weights_only=False)
        keys = data["keys"]
        values = data["values"]
        store = cls(keys=keys, values=values, device=device)
        if "faiss_index" in data:
            store._faiss_index = data["faiss_index"]  # Keep for optional use
        return store


class LawhCrossAttention(nn.Module):
    """
    Trainable cross-attention from hidden state into pre-retrieved Lawḥ (K, V).
    Receives only the K retrieved key/value vectors (e.g. K=256 or 512), not the full store.
    """

    def __init__(
        self,
        d_model: int,
        d_key: int,
        d_value: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        d_value = d_value or d_key
        self.d_value = d_value

        self.query_proj = nn.Linear(d_model, d_key)
        # Always use out_proj so we can zero-init it; at init residual is 0 (avoids Phase 3 loss spike)
        self.out_proj = nn.Linear(d_value, d_model)
        self.norm = build_norm("layernorm", d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        h: torch.Tensor,
        K_batch: torch.Tensor,
        V_batch: torch.Tensor,
        return_metrics: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        """
        h: (B, T, d_model)
        K_batch: (K, d_key), V_batch: (K, d_value) — pre-retrieved for this batch
        Returns: (B, T, d_model) residual output; optional metrics dict.
        """
        B, T, _ = h.shape
        Q = self.query_proj(h)  # (B, T, d_key)
        scale = math.sqrt(self.d_key)
        attn = (Q @ K_batch.t()) / scale  # (B, T, K)
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ V_batch  # (B, T, d_value)
        if self.out_proj is not None:
            out = self.out_proj(out)
        out = self.norm(h + out)

        if not return_metrics:
            return out, None
        with torch.no_grad():
            entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum(dim=-1).mean().item()
            top1_sim = attn_weights.max(dim=-1).values.mean().item()
        return out, {"lawh_attention_entropy": entropy, "lawh_retrieval_similarity": top1_sim}
