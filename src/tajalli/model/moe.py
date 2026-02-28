"""Adversarial Paired MoE (99 Names): 14 experts in 7 complementary pairs.

Optimized: all expert computation via batched einsum — zero Python loops in forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 7 complementary pairs (14 experts). Each tuple is (name_a, name_b) for pair index 0..6.
NAME_PAIRS: list[tuple[str, str]] = [
    ("Al-Qābiḍ", "Al-Bāsiṭ"),   # Pair 0: The Constrictor / The Expander
    ("Al-Muʿizz", "Al-Muḏill"),  # Pair 1: The Honourer / The Humiliator
    ("Al-Muḥyī", "Al-Mumīt"),   # Pair 2: The Giver of Life / The Bringer of Death
    ("Al-Ẓāhir", "Al-Bāṭin"),   # Pair 3: The Manifest / The Hidden
    ("Al-Awwal", "Al-Ākhir"),   # Pair 4: The First / The Last
    ("Al-Laṭīf", "Al-Khabīr"),  # Pair 5: The Subtle / The All-Aware
    ("Al-Jāmiʿ", "Al-Māniʿ"),   # Pair 6: The Gatherer / The Preventer
]
N_EXPERTS = 14  # Default; use n_experts param for configurable count
N_PAIRS = 7


class WisdomRouter(nn.Module):
    """Routes over 14 experts; top-k=2. Returns weights, indices, load_balance_loss, aux."""

    def __init__(self, d_model: int, n_experts: int = N_EXPERTS, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts)

    def forward(
        self, h: torch.Tensor, mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            h: (B, T, d_model)
        Returns:
            weights: (B, T, top_k) — normalized routing weights
            indices: (B, T, top_k) — selected expert indices
            load_balance_loss: scalar
            aux: dict with router_probs, pair_activation_stats
        """
        B, T, _ = h.shape
        logits = self.gate(h)  # (B, T, n_experts)
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # Load balance loss (Switch Transformer style)
        expert_usage = probs.mean(dim=(0, 1))
        load_balance_loss = self.n_experts * (expert_usage * expert_usage).sum()

        # Pair co-activation stats (no Python loop — fully vectorized)
        # For each pair, check if both members appear in top-k for the same token
        n_pairs = self.n_experts // 2
        pair_a_indices = torch.arange(0, self.n_experts, 2, device=h.device)
        pair_b_indices = torch.arange(1, self.n_experts, 2, device=h.device)
        expanded_indices = topk_indices.unsqueeze(-1)  # (B, T, top_k, 1)
        a_present = (expanded_indices == pair_a_indices).any(dim=2)  # (B, T, n_pairs)
        b_present = (expanded_indices == pair_b_indices).any(dim=2)  # (B, T, n_pairs)
        pair_activation = (a_present & b_present).float().mean(dim=(0, 1))  # (N_PAIRS,)

        aux = {
            "pair_activation_stats": pair_activation,
            "router_probs": probs,
            "expert_indices": topk_indices,
            "expert_weights": topk_weights,
        }
        return topk_weights, topk_indices, load_balance_loss, aux


class PairedMoELayer(nn.Module):
    """
    n_experts experts in n_experts//2 pairs + WisdomRouter. All expert computation is batched via einsum.
    No Python loops in forward or run_all_experts.
    d_ff should match Phase 1 FFN so experts can be initialized from it.

    Forward returns (out, aux_dict); out = h + moe_out (residual).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        top_k: int = 2,
        ortho_subset_frac: float = 0.1,
        n_experts: int = N_EXPERTS,
        expert_choice_routing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else (2 * d_model)
        self.n_experts = n_experts
        self.top_k = top_k
        self.ortho_subset_frac = ortho_subset_frac
        self.expert_choice_routing = expert_choice_routing

        self.router = WisdomRouter(d_model, n_experts=n_experts, top_k=top_k)

        # Batched expert weights — all n_experts as stacked tensors
        # w1: (n_experts, d_model, d_ff), w2: (n_experts, d_ff, d_model)
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, self.d_ff))
        self.b1 = nn.Parameter(torch.zeros(n_experts, self.d_ff))
        self.w2 = nn.Parameter(torch.empty(n_experts, self.d_ff, d_model))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))

        # Initialize each expert slice with kaiming
        for i in range(n_experts):
            nn.init.kaiming_uniform_(self.w1[i])
            nn.init.kaiming_uniform_(self.w2[i])

        # Store pair info for external use
        self.pair_indices_a = list(range(0, n_experts, 2))
        self.pair_indices_b = list(range(1, n_experts, 2))

    def _run_all_experts_batched(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run all experts on input via einsum. No loops.

        Args:
            x: (..., d_model) — any leading dims
        Returns:
            out: (..., n_experts, d_model)
        """
        # x: (..., D) -> (..., E, D_ff) -> (..., E, D)
        h = torch.einsum('...d,edf->...ef', x, self.w1) + self.b1  # (..., n_experts, d_ff)
        h = F.gelu(h)
        h = torch.einsum('...ef,efd->...ed', h, self.w2) + self.b2  # (..., 14, d_model)
        return h

    def run_all_experts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Public API for orthogonality loss computation.

        Args:
            x: (B, T, d_model) or (N, d_model)
        Returns:
            out: (B, T, n_experts, d_model) or (N, n_experts, d_model)
        """
        return self._run_all_experts_batched(x)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            h: (B, T, d_model)
            mask: optional attention mask (unused by MoE, passed to router)
            return_aux: include routing stats in output
            **kwargs: ignored (router_step, rescue_experts for API compatibility)
        Returns:
            out: (B, T, d_model) — gated MoE output (residual: h + gate * moe_out)
            aux_dict: routing stats, load_balance_loss, etc.
        """
        B, T, D = h.shape
        N = B * T

        # Run ALL experts on ALL tokens via einsum — single batched op
        all_expert_out = self._run_all_experts_batched(h)  # (B, T, n_experts, D)

        if self.expert_choice_routing:
            # Expert-choice: each expert picks top-C tokens by router weight; capacity C = (B*T*top_k)/n_experts
            logits = self.router.gate(h)  # (B, T, n_experts)
            probs = F.softmax(logits, dim=-1)
            C = max(1, (N * self.top_k) // self.n_experts)
            cap = min(C, N)
            probs_flat = probs.reshape(N, self.n_experts)
            topk_vals, topk_idx = torch.topk(probs_flat, cap, dim=0)  # (cap, n_experts) each
            contribution_mask = torch.zeros(N, self.n_experts, device=h.device, dtype=h.dtype)
            contribution_mask[topk_idx, torch.arange(self.n_experts, device=h.device)] = topk_vals
            moe_out = (all_expert_out.reshape(N, self.n_experts, D) * contribution_mask.unsqueeze(-1)).sum(dim=1)
            moe_out = moe_out.view(B, T, D)
            out = h + moe_out
            load_balance_loss = torch.tensor(0.0, device=h.device, dtype=h.dtype)
            # Pair activation: for each pair, fraction of tokens where both experts contributed
            contrib = contribution_mask.view(B, T, self.n_experts)
            n_pairs = self.n_experts // 2
            pair_a = torch.arange(0, self.n_experts, 2, device=h.device)
            pair_b = torch.arange(1, self.n_experts, 2, device=h.device)
            a_contrib = (contrib[:, :, pair_a] > 0).float().sum(dim=2)  # (B, T)
            b_contrib = (contrib[:, :, pair_b] > 0).float().sum(dim=2)
            pair_activation = torch.stack([
                ((contrib[:, :, 2*p] > 0) & (contrib[:, :, 2*p+1] > 0)).float().mean()
                for p in range(n_pairs)
            ])
            router_aux = {
                "pair_activation_stats": pair_activation,
                "router_probs": probs,
                "expert_indices": contribution_mask.view(B, T, self.n_experts),
                "expert_weights": contribution_mask.view(B, T, self.n_experts),
            }
        else:
            # Token-choice: each token picks top-k experts
            weights, indices, load_balance_loss, router_aux = self.router(h, mask)
            idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, D)
            selected_out = torch.gather(all_expert_out, 2, idx_expanded)
            moe_out = (selected_out * weights.unsqueeze(-1)).sum(dim=2)
            out = h + moe_out

        aux_dict = {}
        if return_aux:
            aux_dict = {
                "load_balance_loss": load_balance_loss,
                "pair_activation_stats": router_aux["pair_activation_stats"],
                "expert_indices": router_aux["expert_indices"],
                "expert_weights": router_aux["expert_weights"],
                "router_probs": router_aux["router_probs"],
                "expert_outputs_subset": all_expert_out.reshape(B * T, self.n_experts, D),
                "all_expert_outputs": all_expert_out,
            }
        return out, aux_dict
