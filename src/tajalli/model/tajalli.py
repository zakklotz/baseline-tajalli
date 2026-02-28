"""Tajallī regeneration: re-projection from frozen essence at each recursive step."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, TYPE_CHECKING, List

from .attention import MultiHeadAttention  # keep attention for now
from tajalli.nncore_mlp import TajalliFFN, build_norm


if TYPE_CHECKING:
    from .moe import PairedMoELayer
    from .lawh import LawhCrossAttention


N_ATTRIBUTES = 7
MAX_TAJALLI_STEPS = 32


class AttributeHeadSimple(nn.Module):
    """
    One attribute head for matrix essence. Computes per-token weights over essence rows,
    then weighted sum. Produces d_model-dim signal from (B, n_rows, d_model) essence.
    """

    def __init__(self, d_model: int, n_essence_rows: int):
        super().__init__()
        self.row_weights = nn.Linear(d_model, n_essence_rows)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor, essence: torch.Tensor) -> torch.Tensor:
        """
        h: (B, S, d_model)
        essence: (B, n_rows, d_model)
        returns: (B, S, d_model)
        """
        weights = F.softmax(self.row_weights(h), dim=-1)
        combined = torch.einsum("bsn,bnd->bsd", weights, essence)
        return self.out_proj(combined)


def _step_to_family_table(n_families: int, family_steps: list[int]) -> list[int]:
    """Build step_to_family[s] = family index for step s. family_steps[f] = first step of family f."""
    table = []
    for s in range(MAX_TAJALLI_STEPS):
        f = 0
        for i in range(1, n_families):
            if s >= family_steps[i]:
                f = i
        table.append(f)
    return table


class TajalliLayer(nn.Module):
    """
    Projects essence through n_attributes heads, gates by previous hidden state,
    produces tajalli_signal. No residual — essence re-projection is the anchor.
    When essence_type=="vector": linear heads Linear(d_essence, d_model) or 2-layer MLP.
    When essence_type=="matrix": AttributeHeadSimple (h → row weights → weighted sum).
    Optional depth_families: step-specific attribute families (e.g. steps 0-2, 3-5, 6+).
    """

    def __init__(
        self,
        d_model: int,
        d_essence: int,
        n_attributes: int = N_ATTRIBUTES,
        d_attr_hidden: Optional[int] = None,
        nonlinear_gate: bool = False,
        essence_type: Literal["vector", "matrix"] = "vector",
        n_essence_rows: int = 64,
        alpha_schedule: Optional[list[float]] = None,
        depth_families: Optional[int] = None,
        family_steps: Optional[list[int]] = None,
        hypernetwork_attributes: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_essence = d_essence
        self.n_attributes = n_attributes
        self.essence_type = essence_type
        self.n_essence_rows = n_essence_rows
        self.hypernetwork_attributes = hypernetwork_attributes
        self.depth_families = depth_families or 0
        self.step_to_family = _step_to_family_table(
            depth_families or 1,
            family_steps or [0],
        ) if (depth_families and family_steps) else None

        def _make_heads():
            if essence_type == "matrix":
                return nn.ModuleList(
                    [AttributeHeadSimple(d_model, n_essence_rows) for _ in range(n_attributes)]
                )
            elif d_attr_hidden is None:
                return nn.ModuleList([nn.Linear(d_essence, d_model) for _ in range(n_attributes)])
            else:
                return nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_essence, d_attr_hidden),
                        nn.GELU(),
                        nn.Linear(d_attr_hidden, d_model),
                    )
                    for _ in range(n_attributes)
                ])

        if self.hypernetwork_attributes:
            essence_dim = d_model if essence_type == "matrix" else d_essence
            self.hyper_net = nn.Sequential(
                nn.Linear(essence_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, n_attributes * d_model),
            )
            self.attribute_heads = None
            self.attribute_heads_family = None
            if nonlinear_gate:
                self.gate = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, n_attributes),
                )
            else:
                self.gate = nn.Linear(d_model, n_attributes)
            self.gate_family = None
        elif self.depth_families and self.step_to_family is not None:
            self.attribute_heads_family = nn.ModuleList([_make_heads() for _ in range(self.depth_families)])
            self.attribute_heads = None
            self.hyper_net = None
            if nonlinear_gate:
                self.gate_family = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_model, d_model // 2),
                        nn.GELU(),
                        nn.Linear(d_model // 2, n_attributes),
                    )
                    for _ in range(self.depth_families)
                ])
            else:
                self.gate_family = nn.ModuleList([nn.Linear(d_model, n_attributes) for _ in range(self.depth_families)])
            self.gate = None
        else:
            self.attribute_heads = _make_heads()
            self.attribute_heads_family = None
            self.hyper_net = None
            if nonlinear_gate:
                self.gate = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, n_attributes),
                )
            else:
                self.gate = nn.Linear(d_model, n_attributes)
            self.gate_family = None
        # Alpha init: optional schedule (e.g. lower at step 2 to reduce over-correction), or default linspace
        init_values = torch.linspace(-0.5, -1.2, MAX_TAJALLI_STEPS)
        if alpha_schedule is not None:
            for i, p in enumerate(alpha_schedule):
                if i >= MAX_TAJALLI_STEPS:
                    break
                p_clamped = max(1e-6, min(1 - 1e-6, p))
                init_values[i] = float(torch.logit(torch.tensor(p_clamped)))
        self.alpha_per_step = nn.Parameter(init_values.clone())

    def forward(
        self,
        essence: torch.Tensor,
        h_prev: torch.Tensor,
        step_idx: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            essence: (B, d_essence) or (B, n_slots, d_essence) for vector;
                     (B, n_rows, d_model) for matrix
            h_prev: (B, seq_len, d_model)
            step_idx: recursive step index (0..MAX_TAJALLI_STEPS-1)
        Returns:
            tajalli_signal: (B, seq_len, d_model)
            metrics: dict with alpha_value, attribute_gate_entropy, attribute_gate_mean
        """
        step_idx_clamped = min(step_idx, MAX_TAJALLI_STEPS - 1)
        if self.hyper_net is not None:
            if self.essence_type == "matrix":
                E = essence.mean(dim=1)  # (B, d_model)
            else:
                E = essence.mean(dim=1) if essence.dim() == 3 else essence  # (B, d_essence)
            mod = self.hyper_net(E).view(-1, self.n_attributes, self.d_model)  # (B, n_attributes, d_model)
            attributes = [h_prev * mod[:, j:j+1, :] for j in range(self.n_attributes)]
            gate_logits = self.gate(h_prev)
        elif self.step_to_family is not None and self.attribute_heads_family is not None:
            family = self.step_to_family[step_idx_clamped]
            heads = self.attribute_heads_family[family]
            if self.essence_type == "matrix":
                attributes = [head(h_prev, essence) for head in heads]
            else:
                if essence.dim() == 3:
                    essence = essence.mean(dim=1)
                B, T, _ = h_prev.shape
                essence_expanded = essence.unsqueeze(1).expand(-1, T, -1)
                attributes = [head(essence_expanded) for head in heads]
            gate_logits = self.gate_family[family](h_prev)
        else:
            if self.essence_type == "matrix":
                attributes = [head(h_prev, essence) for head in self.attribute_heads]
            else:
                if essence.dim() == 3:
                    essence = essence.mean(dim=1)  # (B, d_essence)
                B, T, _ = h_prev.shape
                essence_expanded = essence.unsqueeze(1).expand(-1, T, -1)
                attributes = [head(essence_expanded) for head in self.attribute_heads]
            gate_logits = self.gate(h_prev)
        gate_weights = F.softmax(gate_logits, dim=-1)
        tajalli_signal = sum(
            g * a
            for g, a in zip(
                gate_weights.split(1, dim=-1),
                attributes,
            )
        )
        gate_entropy = -(
            gate_weights * (gate_weights + 1e-10).log()
        ).sum(dim=-1).mean()
        alpha = torch.sigmoid(self.alpha_per_step[step_idx_clamped])
        metrics = {
            "alpha_value": alpha.item(),
            "alpha_step_idx": step_idx_clamped,
            "attribute_gate_entropy": gate_entropy.item(),
            "attribute_gate_mean": gate_weights.mean(dim=(0, 1)).detach().cpu(),
            "_gate_entropy_tensor": gate_entropy,  # live tensor for regularization loss
        }
        return tajalli_signal, metrics


class TajalliBlock(nn.Module):
    """
    One recursive step: TajalliLayer + MultiHeadAttention + FFN (and optionally MoE).
    h_t = alpha * tajalli_signal + (1 - alpha) * f(h_{t-1})
    Weights shared across recursive steps.

    When moe_layer is present: use only MoE (no FFN). f_h = norm_ffn(f_h + moe_out).
    When absent: standard FFN path f_h = norm_ffn(f_h + ffn(f_h)).
    """

    def __init__(
        self,
        d_model: int,
        d_essence: int,
        n_heads: int,
        d_head: int,
        d_ff: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        moe_layer: Optional[object] = None,  # PairedMoELayer
        lawh_cross_attn: Optional["LawhCrossAttention"] = None,
        n_attributes: int = N_ATTRIBUTES,
        d_attr_hidden: Optional[int] = None,
        nonlinear_gate: bool = False,
        essence_type: Literal["vector", "matrix"] = "vector",
        n_essence_rows: int = 64,
        alpha_schedule: Optional[list[float]] = None,
        n_inner_layers: int = 0,
        depth_families: Optional[int] = None,
        family_steps: Optional[list[int]] = None,
        hypernetwork_attributes: bool = False,
    ):
        super().__init__()
        self.tajalli_layer = TajalliLayer(
            d_model,
            d_essence,
            n_attributes=n_attributes,
            d_attr_hidden=d_attr_hidden,
            nonlinear_gate=nonlinear_gate,
            essence_type=essence_type,
            n_essence_rows=n_essence_rows,
            alpha_schedule=alpha_schedule,
            depth_families=depth_families,
            family_steps=family_steps,
            hypernetwork_attributes=hypernetwork_attributes,
        )
        self.attention = MultiHeadAttention(
            d_model, n_heads, d_head, dropout, max_seq_len
        )
        self.lawh_cross_attn = lawh_cross_attn
        self.moe_layer = moe_layer
        self.ffn = TajalliFFN(d_model=d_model, d_ff=d_ff, dropout=dropout) if moe_layer is None else None
        self.norm_attn = build_norm("layernorm", d_model)
        self.norm_ffn = build_norm("layernorm", d_model)
        # Extra transformer blocks per recursive step — unique weights, shared across steps.
        # Each adds one full attention+FFN pass, doubling compute per step when n_inner_layers=1.
        self.inner_attns = nn.ModuleList([
            MultiHeadAttention(d_model, n_heads, d_head, dropout, max_seq_len)
            for _ in range(n_inner_layers)
        ])
        self.inner_ffns = nn.ModuleList([
            TajalliFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
            for _ in range(n_inner_layers)
        ])
        self.inner_norms_attn = nn.ModuleList([build_norm("layernorm", d_model) for _ in range(n_inner_layers)])
        self.inner_norms_ffn = nn.ModuleList([build_norm("layernorm", d_model) for _ in range(n_inner_layers)])

    def forward(
        self,
        h: torch.Tensor,
        essence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        lawh_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_lawh_this_step: bool = False,
        step_idx: int = 0,
        memory_h: Optional[torch.Tensor] = None,
        mem_pos_start: int = 0,
        cached_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv: bool = False,
    ) -> tuple:
        """
        Args:
            h: (B, seq_len, d_model)
            essence: (B, d_essence)
            mask: optional causal/attention mask
            lawh_kv: optional (K_keys, K_values) for Lawḥ cross-attention
            use_lawh_this_step: if True and lawh_cross_attn + lawh_kv present, run Lawḥ
            step_idx: recursive step index for per-step alpha
            cached_kv: optional (K, V) from step 0 for recursive KV sharing
            return_kv: if True and cached_kv is None, return (h_new, metrics, (K, V))
        Returns:
            h_new: (B, seq_len, d_model)
            metrics: if return_metrics, dict; else None
            optional: (K, V) when return_kv and cached_kv was None
        """
        h_prev = h
        tajalli_signal, tajalli_metrics = self.tajalli_layer(essence, h_prev, step_idx=step_idx)
        attn_out = self.attention(
            h_prev, mask,
            memory_h=memory_h,
            mem_pos_start=mem_pos_start,
            cached_kv=cached_kv,
            return_kv=return_kv and (memory_h is None),
        )
        if isinstance(attn_out, tuple):
            attn_out, kv_out = attn_out
        else:
            kv_out = None
        f_h = self.norm_attn(h_prev + attn_out)

        if use_lawh_this_step and self.lawh_cross_attn is not None and lawh_kv is not None:
            K_batch, V_batch = lawh_kv
            f_h, lawh_metrics = self.lawh_cross_attn(
                f_h, K_batch, V_batch, return_metrics=return_metrics
            )
        else:
            lawh_metrics = None

        if self.moe_layer is not None:
            moe_out, moe_aux = self.moe_layer(f_h, mask=mask, return_aux=return_metrics)
            f_h = self.norm_ffn(f_h + moe_out)
        else:
            moe_aux = None
            f_h = self.norm_ffn(f_h + self.ffn(f_h))

        step_idx_clamped = min(step_idx, MAX_TAJALLI_STEPS - 1)
        alpha = torch.sigmoid(self.tajalli_layer.alpha_per_step[step_idx_clamped])
        # h_new = alpha * essence + (1-alpha) * transformer; alpha in [0,1] blends toward essence
        h_new = alpha * tajalli_signal + (1 - alpha) * f_h

        # Additional compute blocks per recursive step (n_inner_layers > 0)
        for inner_attn, inner_ffn, inner_norm_a, inner_norm_f in zip(
            self.inner_attns, self.inner_ffns, self.inner_norms_attn, self.inner_norms_ffn
        ):
            h_new = inner_norm_a(h_new + inner_attn(h_new, mask, memory_h=memory_h, mem_pos_start=mem_pos_start))
            h_new = inner_norm_f(h_new + inner_ffn(h_new))

        if return_metrics:
            with torch.no_grad():
                drift_cos = F.cosine_similarity(
                    h_new.flatten(1), h_prev.flatten(1), dim=1
                ).mean()
                hidden_norm = h_new.norm(dim=-1).mean()
                essence_anchoring = F.cosine_similarity(
                    h_new.flatten(1),
                    tajalli_signal.flatten(1),
                    dim=1,
                ).mean()
            metrics = {
                "drift_cosine_sim": drift_cos.item(),
                "hidden_norm": hidden_norm.item(),
                "essence_anchoring": essence_anchoring.item(),
                **tajalli_metrics,
            }
            if moe_aux is not None:
                metrics["moe_aux"] = moe_aux
            if lawh_metrics is not None:
                metrics.update(lawh_metrics)
            if kv_out is not None:
                return h_new, metrics, kv_out
            return h_new, metrics
        if kv_out is not None:
            return h_new, None, kv_out
        return h_new, None


def _resolve_lawh_at_steps(lawh_at_steps: Optional[List[int]], n_steps: int) -> set:
    """Convert lawh_at_steps (e.g. [0, -1]) to set of step indices in [0, n_steps)."""
    if not lawh_at_steps:
        return set()
    resolved = set()
    for s in lawh_at_steps:
        if s < 0:
            resolved.add(n_steps + s)
        else:
            resolved.add(s)
    return {i for i in resolved if 0 <= i < n_steps}


class TajalliStack(nn.Module):
    """
    Applies TajalliBlock (or TajalliBlockV2 with Jamba) for N recursive steps. Shared weights.
    h_0 = token_embeddings; h_t = block(h_{t-1}, essence)
    Lawḥ cross-attention runs only at steps in lawh_at_steps (e.g. first and last).
    """

    def __init__(
        self,
        d_model: int,
        d_essence: int,
        n_heads: int,
        d_head: int,
        d_ff: int,
        n_steps: int = 6,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        use_moe: bool = False,
        ortho_subset_frac: float = 0.1,
        lawh_cross_attn: Optional["LawhCrossAttention"] = None,
        lawh_at_steps: Optional[List[int]] = None,
        use_jamba: bool = False,
        jamba_config: Optional[dict] = None,
        n_experts: int = 14,
        n_attributes: int = N_ATTRIBUTES,
        d_attr_hidden: Optional[int] = None,
        nonlinear_gate: bool = False,
        essence_type: Literal["vector", "matrix"] = "vector",
        n_essence_rows: int = 64,
        alpha_schedule: Optional[list[float]] = None,
        n_inner_layers: int = 0,
        exit_router: Optional[object] = None,  # ExitRouter for token-level adaptive recursion
        use_recursive_kv_cache: bool = False,
        expert_choice_routing: bool = False,
        depth_families: Optional[int] = None,
        family_steps: Optional[list[int]] = None,
        hypernetwork_attributes: bool = False,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.use_moe = use_moe
        self.lawh_at_steps = lawh_at_steps or []
        self.exit_router = exit_router
        self.use_recursive_kv_cache = use_recursive_kv_cache
        tajalli_kw = dict(
            n_attributes=n_attributes,
            d_attr_hidden=d_attr_hidden,
            nonlinear_gate=nonlinear_gate,
            essence_type=essence_type,
            n_essence_rows=n_essence_rows,
            alpha_schedule=alpha_schedule,
            n_inner_layers=n_inner_layers,
            depth_families=depth_families,
            family_steps=family_steps,
            hypernetwork_attributes=hypernetwork_attributes,
        )
        moe_layer = None
        if use_moe:
            from .moe import PairedMoELayer
            moe_layer = PairedMoELayer(
                d_model,
                d_ff=d_ff,
                top_k=2,
                ortho_subset_frac=ortho_subset_frac,
                n_experts=n_experts,
                expert_choice_routing=expert_choice_routing,
            )
        if use_jamba and jamba_config:
            from .jamba_adapter import JambaStyleBlock
            from .tajalli_block_v2 import TajalliBlockV2
            jamba_block = JambaStyleBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_head=d_head,
                d_ff=d_ff,
                dropout=dropout,
                max_seq_len=max_seq_len,
                ortho_subset_frac=ortho_subset_frac,
                mamba_d_state=jamba_config.get("mamba_d_state", 16),
                mamba_dt_rank=jamba_config.get("mamba_dt_rank", 32),
                mamba_expand=jamba_config.get("mamba_expand", 2),
                mamba_d_conv=jamba_config.get("mamba_d_conv", 4),
            )
            self.block = TajalliBlockV2(
                d_model, d_essence, n_heads, d_head, d_ff,
                dropout, max_seq_len, moe_layer=None,
                lawh_cross_attn=lawh_cross_attn,
                jamba_block=jamba_block,
                use_jamba=True,
                **tajalli_kw,
            )
        else:
            self.block = TajalliBlock(
                d_model, d_essence, n_heads, d_head, d_ff,
                dropout, max_seq_len, moe_layer=moe_layer,
                lawh_cross_attn=lawh_cross_attn,
                **tajalli_kw,
            )

    def forward(
        self,
        h: torch.Tensor,
        essence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_step_metrics: bool = False,
        n_steps_override: Optional[int] = None,
        lawh_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        memory_h: Optional[torch.Tensor] = None,
        mem_pos_start: int = 0,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        """
        Args:
            h: (B, seq_len, d_model) — token embeddings
            essence: from EssenceCore.forward(batch_size)
            mask: optional
            n_steps_override: if set, use this many steps instead of self.n_steps
            lawh_kv: optional (K_keys, K_values) — pre-retrieved for this batch; used only at lawh_at_steps
        Returns:
            h: (B, seq_len, d_model) after N steps
            step_metrics: per-step metrics; last step includes moe_aux with all_expert_outputs
        """
        steps = n_steps_override if n_steps_override is not None else self.n_steps
        step_metrics = {} if return_step_metrics else None
        last_moe_aux = None
        gate_entropy_tensors = []
        exit_entropy_tensors = []
        lawh_steps = _resolve_lawh_at_steps(self.lawh_at_steps, steps)

        use_exit_router = self.exit_router is not None
        if use_exit_router:
            h_out = h.clone()
            B, T, D = h.shape
            exited_so_far = torch.zeros(B, T, 1, dtype=torch.bool, device=h.device)

        kv_cache = None
        for step in range(steps):
            # Collect metrics every step when return_step_metrics (for stability eval);
            # moe_aux is large so we only keep it from the last step.
            return_m = return_step_metrics
            use_lawh = step in lawh_steps
            use_kv_cache = self.use_recursive_kv_cache and memory_h is None
            block_out = self.block(
                h, essence, mask,
                return_metrics=return_m,
                lawh_kv=lawh_kv,
                use_lawh_this_step=use_lawh,
                step_idx=step,
                memory_h=memory_h,
                mem_pos_start=mem_pos_start,
                cached_kv=kv_cache,
                return_kv=use_kv_cache and (step == 0),
            )
            if len(block_out) == 3:
                h, m, kv_cache = block_out
            else:
                h, m = block_out

            if use_exit_router:
                scores, exit_mask = self.exit_router(h, exited_so_far)
                # Write exited positions to output buffer; keep rest for next step
                h_out = torch.where(exit_mask.expand_as(h_out), h, h_out)
                exited_so_far = exited_so_far | exit_mask
                # Binary entropy -mean(s*log(s)+(1-s)*log(1-s)) for L_exit (trainer subtracts lambda_exit * this)
                s = scores.clamp(1e-7, 1.0 - 1e-7)
                ent = -(s * s.log() + (1 - s) * (1 - s).log()).mean()
                exit_entropy_tensors.append(ent)

            if return_m and m is not None:
                for k, v in m.items():
                    if k == "moe_aux":
                        last_moe_aux = v
                    elif k == "_gate_entropy_tensor":
                        gate_entropy_tensors.append(v)  # keep live tensor for loss
                    else:
                        step_metrics[f"step_{step}_{k}"] = v

        if use_exit_router:
            h_out = torch.where(~exited_so_far.expand_as(h_out), h, h_out)
            h = h_out
            if exit_entropy_tensors:
                step_metrics = step_metrics if step_metrics is not None else {}
                step_metrics["_exit_entropy_tensor"] = sum(exit_entropy_tensors) / len(exit_entropy_tensors)

        # MoE aux (expert indices, etc.) only from last step to avoid large tensors
        if return_step_metrics and last_moe_aux is not None:
            step_metrics["moe_aux"] = last_moe_aux
        # Sum gate entropy tensors across all recursive steps for regularization loss
        if gate_entropy_tensors:
            step_metrics["gate_entropy_loss"] = sum(gate_entropy_tensors) / len(gate_entropy_tensors)

        return h, step_metrics
