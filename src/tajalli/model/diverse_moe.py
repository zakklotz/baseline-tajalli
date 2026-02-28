"""Diverse Asmā' MoE: top-2 token-choice routing + expert dropout + diversity losses."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .moe import N_EXPERTS, N_PAIRS, NAME_PAIRS

# Mask logit for dropped experts; must fit float16 (max magnitude ~65504)
MASK_LOGIT = -1e4


class TopKTokenRouter(nn.Module):
    """Top-k token-choice router: gate logits, return top-k indices and weights plus full logits."""

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k
        self.n_experts = n_experts

    def forward(
        self, h: torch.Tensor, step: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # h: (N, d_model) -> indices (N, k), weights (N, k), logits (N, n_experts)
        logits = self.gate(h)
        if step is not None:
            temp = max(1.0, 2.0 - step / 5000.0)
            logits = logits / temp
        topk_vals, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)
        return topk_indices, weights, logits


class ExpertFFN(nn.Module):
    """Single expert FFN: same signature as Phase 1 FeedForward (w1, w2) for weight copy."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(
            self.w2(F.gelu(self.w1(x))),
            p=self.dropout,
            training=self.training,
        )


class DiverseAsmaaMoE(nn.Module):
    """
    MoE with top-2 token-choice routing, expert dropout, and diversity losses.
    Forward returns (out, aux_dict); out = h + moe_out (residual).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        n_experts: int = N_EXPERTS,
        top_k: int = 2,
        expert_dropout_rate: float = 0.1,
        entropy_weight: float = 0.01,
        ortho_subset_frac: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else (2 * d_model)
        self.n_experts = n_experts
        self.ortho_subset_frac = ortho_subset_frac
        self.expert_dropout_rate = expert_dropout_rate
        self.entropy_weight = entropy_weight

        self.router = TopKTokenRouter(d_model, n_experts=n_experts, top_k=top_k)
        self.pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]

        # Batched expert weights (same layout as PairedMoELayer) for fast forward
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, self.d_ff))
        self.b1 = nn.Parameter(torch.zeros(n_experts, self.d_ff))
        self.w2 = nn.Parameter(torch.empty(n_experts, self.d_ff, d_model))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))
        for i in range(n_experts):
            nn.init.kaiming_uniform_(self.w1[i])
            nn.init.kaiming_uniform_(self.w2[i])

    def _run_all_experts(self, x: torch.Tensor) -> torch.Tensor:
        """Run all experts on x via batched einsum. x: (N, d_model) -> out: (N, n_experts, d_model)."""
        h = torch.einsum("nd,edf->nef", x, self.w1) + self.b1
        h = F.gelu(h)
        return torch.einsum("nef,efd->ned", h, self.w2) + self.b2

    def compute_diversity_losses(self, h: torch.Tensor) -> dict:
        """Compute entropy and ortho losses for diversity."""
        h_flat = h.reshape(-1, h.shape[-1])
        N = h_flat.shape[0]

        # 1. Router entropy loss
        scores = self.router.gate(h_flat)  # (N, n_experts)
        probs = F.softmax(scores, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        max_entropy = math.log(self.n_experts)
        entropy_loss = (max_entropy - entropy) / max_entropy

        # 2. Pair orthogonality on 10% subset (use batched expert output)
        sample_size = max(1, N // 10)
        sample_idx = torch.randperm(N, device=h.device)[:sample_size]
        sample = h_flat[sample_idx]
        all_out = self._run_all_experts(sample)  # (sample_size, n_experts, d_model)
        ortho_loss = torch.tensor(0.0, device=h.device, dtype=h.dtype)
        for a, b in self.pairs:
            out_a = all_out[:, a, :]
            out_b = all_out[:, b, :]
            cos_sim = F.cosine_similarity(out_a, out_b, dim=-1).mean()
            ortho_loss = ortho_loss + cos_sim.abs()
        ortho_loss = ortho_loss / len(self.pairs)

        return {"entropy_loss": entropy_loss, "ortho_loss": ortho_loss}

    def _forward_token_choice(
        self,
        h: torch.Tensor,
        h_flat: torch.Tensor,
        B: int,
        S: int,
        D: int,
        return_aux: bool,
        expert_mask: Optional[torch.Tensor] = None,
        router_step: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Top-2 token-choice with optional expert dropout (mask dropped experts' logits)."""
        _, _, logits = self.router(h_flat, step=router_step)  # (N, n_experts)
        if expert_mask is not None:
            logits = logits.masked_fill(~expert_mask.unsqueeze(0), MASK_LOGIT)
        probs = F.softmax(logits, dim=-1)
        top2_probs, top2_indices = torch.topk(probs, 2, dim=-1)  # (N, 2)
        top2_weights = top2_probs / top2_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        all_out = self._run_all_experts(h_flat)  # (N, n_experts, D)
        idx0 = top2_indices[:, 0]
        idx1 = top2_indices[:, 1]
        out0 = all_out[torch.arange(h_flat.shape[0], device=h.device), idx0]
        out1 = all_out[torch.arange(h_flat.shape[0], device=h.device), idx1]
        moe_out = top2_weights[:, 0:1] * out0 + top2_weights[:, 1:2] * out1
        result = (h_flat + moe_out).reshape(B, S, D)
        result = torch.clamp(result, -100.0, 100.0)
        aux_dict = {}
        if return_aux:
            pair_activation = self._pair_activation_from_top2(top2_indices)
            diversity = self.compute_diversity_losses(h)
            aux_dict = {
                "load_balance_loss": torch.tensor(0.0, device=h.device),
                "router_probs": probs,
                "expert_indices": top2_indices.reshape(B, S, 2),
                "expert_weights": top2_weights,
                "entropy_loss": diversity["entropy_loss"],
                "ortho_loss": diversity["ortho_loss"],
                "expert_outputs_subset": all_out,
                "pair_activation_stats": pair_activation,
            }
        return result, aux_dict

    def _pair_activation_from_top2(self, top2_indices: torch.Tensor) -> torch.Tensor:
        """Fraction of tokens where both members of each pair appear in top-2. (N_PAIRS,)"""
        N = top2_indices.shape[0]
        pair_activation = torch.zeros(len(self.pairs), device=top2_indices.device)
        for p, (a, b) in enumerate(self.pairs):
            has_a = (top2_indices == a).any(dim=1).float()
            has_b = (top2_indices == b).any(dim=1).float()
            pair_activation[p] = (has_a * has_b).mean()
        return pair_activation

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
        router_step: Optional[int] = None,
        rescue_experts: Optional[list] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Top-2 token-choice routing with expert dropout and diversity losses.
        No expert-choice loop; batched expert forward for speed.
        """
        B, S, D = h.shape
        h_flat = h.reshape(B * S, D)

        expert_mask = None
        if self.training and self.expert_dropout_rate > 0:
            expert_mask = (torch.rand(self.n_experts, device=h.device) > self.expert_dropout_rate)
            for a, b in self.pairs:
                if not expert_mask[a].item() and not expert_mask[b].item():
                    if torch.rand(1, device=h.device).item() > 0.5:
                        expert_mask[a] = True
                    else:
                        expert_mask[b] = True

        return self._forward_token_choice(h, h_flat, B, S, D, return_aux, expert_mask=expert_mask, router_step=router_step)


class RandomAssignMoE(nn.Module):
    """
    Single-expert MoE with random assignment during training.
    Each token is assigned to exactly one expert (random in training, router argmax at inference).
    Router produces a sigmoid weight for the assigned expert only. Returns residual (expert contribution).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        n_experts: int = N_EXPERTS,
        ortho_subset_frac: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else (2 * d_model)
        self.n_experts = n_experts
        self.ortho_subset_frac = ortho_subset_frac
        self.pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, self.d_ff))
        self.b1 = nn.Parameter(torch.zeros(n_experts, self.d_ff))
        self.w2 = nn.Parameter(torch.empty(n_experts, self.d_ff, d_model))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))
        for i in range(n_experts):
            nn.init.kaiming_uniform_(self.w1[i])
            nn.init.kaiming_uniform_(self.w2[i])

    def _run_all_experts(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, d_model) -> (N, n_experts, d_model)."""
        h = torch.einsum("nd,edf->nef", x, self.w1) + self.b1
        h = F.gelu(h)
        return torch.einsum("nef,efd->ned", h, self.w2) + self.b2

    def _compute_ortho_loss(self, h_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Ortho loss on subset; returns (ortho_loss, expert_outputs_subset)."""
        N = h_flat.shape[0]
        sample_size = max(1, int(N * self.ortho_subset_frac))
        sample_idx = torch.randperm(N, device=h_flat.device)[:sample_size]
        sample = h_flat[sample_idx]
        all_out = self._run_all_experts(sample)
        ortho_loss = torch.tensor(0.0, device=h_flat.device, dtype=h_flat.dtype)
        for a, b in self.pairs:
            out_a = all_out[:, a, :]
            out_b = all_out[:, b, :]
            cos_sim = F.cosine_similarity(out_a, out_b, dim=-1).mean()
            ortho_loss = ortho_loss + cos_sim.abs()
        ortho_loss = ortho_loss / len(self.pairs)
        return ortho_loss, all_out

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
        router_step: Optional[int] = None,
        rescue_experts: Optional[list] = None,
    ) -> tuple[torch.Tensor, dict]:
        B, S, D = h.shape
        N = B * S
        h_flat = h.reshape(N, D)
        logits = self.router(h_flat)

        if self.training:
            assignments = torch.randint(0, self.n_experts, (N,), device=h.device, dtype=torch.long)
        else:
            assignments = logits.argmax(dim=-1)

        weights = torch.sigmoid(logits.gather(1, assignments.unsqueeze(1)).squeeze(1))
        all_out = self._run_all_experts(h_flat)
        moe_out_flat = all_out[torch.arange(N, device=h.device), assignments, :] * weights.unsqueeze(-1)
        moe_out = moe_out_flat.reshape(B, S, D)

        aux_dict = {}
        if return_aux:
            ortho_loss, expert_outputs_subset = self._compute_ortho_loss(h_flat)
            expert_out_norms = torch.zeros(self.n_experts, device=h.device, dtype=h.dtype)
            for e in range(self.n_experts):
                em = assignments == e
                if em.any():
                    expert_out_norms[e] = all_out[em, e, :].float().norm(dim=-1).mean()
            aux_dict = {
                "load_balance_loss": torch.tensor(0.0, device=h.device),
                "router_probs": logits,
                "expert_indices": assignments.reshape(B, S, 1),
                "router_weights": weights,
                "ortho_loss": ortho_loss,
                "expert_outputs_subset": expert_outputs_subset,
                "expert_out_norms": expert_out_norms,
            }
        return moe_out, aux_dict


class TwoPhaseMoE(nn.Module):
    """
    Two-phase MoE: Phase A (random assignment), transition (blend), Phase B (learned routing).
    Returns residual only. Uses router_step for phase logic; softmax weights; expert dropout
    and entropy loss only in Phase B (router_step >= switch_step).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        n_experts: int = N_EXPERTS,
        ortho_subset_frac: float = 0.1,
        switch_step: int = 10000,
        transition_steps: int = 2000,
        expert_dropout_rate: float = 0.15,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else (2 * d_model)
        self.n_experts = n_experts
        self.ortho_subset_frac = ortho_subset_frac
        self.switch_step = switch_step
        self.transition_steps = transition_steps
        self.expert_dropout_rate = expert_dropout_rate
        self.entropy_weight = entropy_weight
        self.pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, self.d_ff))
        self.b1 = nn.Parameter(torch.zeros(n_experts, self.d_ff))
        self.w2 = nn.Parameter(torch.empty(n_experts, self.d_ff, d_model))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))
        for i in range(n_experts):
            nn.init.kaiming_uniform_(self.w1[i])
            nn.init.kaiming_uniform_(self.w2[i])

    def _run_all_experts(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, d_model) -> (N, n_experts, d_model)."""
        h = torch.einsum("nd,edf->nef", x, self.w1) + self.b1
        h = F.gelu(h)
        return torch.einsum("nef,efd->ned", h, self.w2) + self.b2

    def _compute_ortho_loss(self, h_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Ortho loss on subset; returns (ortho_loss, expert_outputs_subset)."""
        N = h_flat.shape[0]
        sample_size = max(1, int(N * self.ortho_subset_frac))
        sample_idx = torch.randperm(N, device=h_flat.device)[:sample_size]
        sample = h_flat[sample_idx]
        all_out = self._run_all_experts(sample)
        ortho_loss = torch.tensor(0.0, device=h_flat.device, dtype=h_flat.dtype)
        for a, b in self.pairs:
            out_a = all_out[:, a, :]
            out_b = all_out[:, b, :]
            cos_sim = F.cosine_similarity(out_a, out_b, dim=-1).mean()
            ortho_loss = ortho_loss + cos_sim.abs()
        ortho_loss = ortho_loss / len(self.pairs)
        return ortho_loss, all_out

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
        router_step: Optional[int] = None,
        rescue_experts: Optional[list] = None,
    ) -> tuple[torch.Tensor, dict]:
        B, S, D = h.shape
        N = B * S
        h_flat = h.reshape(N, D)
        logits = self.router(h_flat)

        # Phase: A (random), transition (blend), B (router)
        if not self.training:
            phase = "B"
            assignments = logits.argmax(dim=-1)
        elif router_step is None or router_step >= self.switch_step + self.transition_steps:
            phase = "B"
            assignments = logits.argmax(dim=-1)
        elif router_step < self.switch_step:
            phase = "A"
            assignments = torch.randint(0, self.n_experts, (N,), device=h.device, dtype=torch.long)
        else:
            phase = "transition"
            p_router = (router_step - self.switch_step) / max(1, self.transition_steps)
            use_router = torch.rand(N, device=h.device, dtype=h.dtype) < p_router
            random_assignments = torch.randint(0, self.n_experts, (N,), device=h.device, dtype=torch.long)
            router_assignments = logits.argmax(dim=-1)
            assignments = torch.where(use_router, router_assignments, random_assignments)

        # Expert dropout (Phase B only): mask logits then re-softmax
        logits_for_probs = logits
        if self.training and router_step is not None and router_step >= self.switch_step and self.expert_dropout_rate > 0:
            expert_mask = (torch.rand(self.n_experts, device=h.device, dtype=h.dtype) > self.expert_dropout_rate)
            for a, b in self.pairs:
                if not expert_mask[a] and not expert_mask[b]:
                    if torch.rand(1, device=h.device).item() > 0.5:
                        expert_mask = expert_mask.clone()
                        expert_mask[a] = True
                    else:
                        expert_mask = expert_mask.clone()
                        expert_mask[b] = True
            logits_for_probs = logits.masked_fill(~expert_mask.unsqueeze(0), MASK_LOGIT)

        probs = F.softmax(logits_for_probs, dim=-1)
        weights = probs.gather(1, assignments.unsqueeze(1)).squeeze(1)

        # Rescue routing (Phase B only): force 5% of tokens to underutilized experts for 200 steps
        if self.training and phase == "B" and rescue_experts and len(rescue_experts) > 0:
            n_rescue = len(rescue_experts)
            n_per_expert = max(1, int(0.05 * N / n_rescue))
            perm = torch.randperm(N, device=h.device)
            assignments = assignments.clone()
            weights = weights.clone()
            start = 0
            for e in rescue_experts:
                end = min(start + n_per_expert, N)
                if start < end:
                    idx_rescue = perm[start:end]
                    assignments[idx_rescue] = e
                    weights[idx_rescue] = 1.0
                    start = end
                if start >= N:
                    break

        all_out = self._run_all_experts(h_flat)
        moe_out_flat = all_out[torch.arange(N, device=h.device), assignments, :] * weights.unsqueeze(-1)
        moe_out = moe_out_flat.reshape(B, S, D)

        aux_dict = {}
        if return_aux:
            ortho_loss, expert_outputs_subset = self._compute_ortho_loss(h_flat)
            expert_out_norms = torch.zeros(self.n_experts, device=h.device, dtype=h.dtype)
            for e in range(self.n_experts):
                emask = assignments == e
                if emask.any():
                    expert_out_norms[e] = all_out[emask, e, :].float().norm(dim=-1).mean()
            aux_dict = {
                "load_balance_loss": torch.tensor(0.0, device=h.device),
                "router_probs": logits,
                "expert_indices": assignments.reshape(B, S, 1),
                "router_weights": weights,
                "ortho_loss": ortho_loss,
                "expert_outputs_subset": expert_outputs_subset,
                "expert_out_norms": expert_out_norms,
                "phase": phase,
            }
            if phase in ("transition", "B") and router_step is not None and router_step >= self.switch_step:
                p = probs + 1e-10
                entropy_per_token = -(p * p.log()).sum(dim=-1)
                aux_dict["entropy_loss"] = entropy_per_token.mean()
            else:
                aux_dict["entropy_loss"] = torch.tensor(0.0, device=h.device, dtype=h.dtype)
        return moe_out, aux_dict

