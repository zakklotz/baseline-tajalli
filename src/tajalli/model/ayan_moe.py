"""Aʿyān Thābitah MoE: soft mixture of all experts weighted by fixed orthogonal archetypes. No router, no collapse."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any


class PolarizedExpert(nn.Module):
    """
    A single expert that produces both positive and negative transformations.
    Implements Jamʿ al-Aḍdād (Union of Opposites) — one expert captures
    both sides of its attribute (e.g., Qābiḍ AND Bāsiṭ from the same weights).

    output = ReLU(W1 @ x + b1) @ W2 - ReLU(-W1 @ x - b1) @ W2

    The positive path is the "manifest" aspect, the negative path is the
    "hidden" aspect. Both are always active. Neither can die because they
    share weights.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.w1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2.weight, a=math.sqrt(5))
        if self.w1.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w1.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.w1.bias, -bound, bound)
        if self.w2.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w2.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.w2.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        pos = F.relu(h)
        neg = F.relu(-h)
        out = self.w2(self.dropout(pos)) - self.w2(self.dropout(neg))
        return out


class StandardExpert(nn.Module):
    """Standard GELU FFN expert (same structure as Phase 1 FeedForward)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class AyanThābitahMoE(nn.Module):
    """
    Soft Mixture of All Experts weighted by fixed orthogonal archetypes.

    7 fixed (non-trainable) orthogonal archetype vectors define directions
    in hidden space. Each token's similarity to each archetype determines
    how much each expert contributes. Every expert always contributes to
    every token — no routing, no selection, no collapse.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_archetypes: int = 7,
        dropout: float = 0.0,
        archetype_temp: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_archetypes = n_archetypes
        self.archetype_temp = archetype_temp

        # Fixed orthogonal archetype vectors (Aʿyān Thābitah)
        raw = torch.randn(n_archetypes, d_model)
        Q, _ = torch.linalg.qr(raw.T)
        archetypes = Q.T
        self.register_buffer("archetypes", archetypes)

        self.experts = nn.ModuleList([
            StandardExpert(d_model, d_ff, dropout)
            for _ in range(n_archetypes)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.expert_names = [
            "Al-Qābiḍ/Al-Bāsiṭ",
            "Al-Muʿizz/Al-Muḍill",
            "Al-Muḥyī/Al-Mumīt",
            "Al-Ẓāhir/Al-Bāṭin",
            "Al-Awwal/Al-Ākhir",
            "Al-Laṭīf/Al-Khabīr",
            "Al-Jāmiʿ/Al-Māniʿ",
        ]

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            h: [batch, seq_len, d_model]
            mask: unused (API compatibility)
            return_aux: if True, include archetype_weights and entropy_loss in aux
        Returns:
            result: [batch, seq_len, d_model] — norm(h + weighted_experts)
            aux_dict: archetype_weights, entropy_loss when return_aux
        """
        B, S, D = h.shape

        step = kwargs.get("router_step", None)
        if step is None or step < 40000:
            temp = self.archetype_temp
        else:
            temp = self.archetype_temp - 0.5 * min(1.0, (step - 40000) / 10000)

        similarities = torch.einsum("bsd,nd->bsn", h, self.archetypes)
        weights = F.softmax(similarities / temp, dim=-1)

        output = torch.zeros_like(h)
        for j, expert in enumerate(self.experts):
            expert_out = expert(h)
            w_j = weights[:, :, j].unsqueeze(-1)
            output = output + w_j * expert_out

        result = self.norm(h + output)

        aux_dict: dict = {}
        if return_aux:
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
            max_entropy = math.log(self.n_archetypes)
            entropy_loss = (max_entropy - entropy) / max_entropy
            aux_dict = {
                "archetype_weights": weights,
                "entropy_loss": entropy_loss,
            }
        return result, aux_dict

    def get_utilization(self, h: torch.Tensor) -> torch.Tensor:
        """Mean weight per archetype over batch and sequence. [n_archetypes]."""
        with torch.no_grad():
            similarities = torch.einsum("bsd,nd->bsn", h, self.archetypes)
            weights = F.softmax(similarities / self.archetype_temp, dim=-1)
            return weights.mean(dim=(0, 1))

    def compute_aux_losses(self, h: torch.Tensor) -> dict:
        """Entropy loss to encourage diversity. For use outside forward if needed."""
        with torch.no_grad():
            similarities = torch.einsum("bsd,nd->bsn", h, self.archetypes)
            weights = F.softmax(similarities / self.archetype_temp, dim=-1)
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
        max_entropy = math.log(self.n_archetypes)
        entropy_loss = (max_entropy - entropy) / max_entropy
        return {"entropy_loss": entropy_loss}
