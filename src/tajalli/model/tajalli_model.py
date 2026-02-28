"""Tajallī Phase 1, Phase 2, and Phase 3 models: embeddings + TajalliStack + output head."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Literal, Optional

from .essence import EssenceCore, EssenceCoreMatrix
from .tajalli import TajalliStack
from .moe import N_EXPERTS  # Default n_experts; config can override
from .lawh import LawhMemoryStore, LawhCrossAttention
from .exit_router import ExitRouter
from .barzakh import BarzakhBottleneck
from .qadr import QadrConstraints
from tajalli.nncore_mlp import build_norm

class TajalliModelPhase1(nn.Module):
    """
    Phase 1 model: TokenEmbedding + EssenceCore + TajalliStack + OutputHead.
    Weight-tied output head with embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_essence: int,
        n_heads: int,
        d_head: int,
        d_ff: int,
        n_steps: int = 6,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        padding_idx: Optional[int] = None,
        essence_init: str = "spectral",
        essence_path: Optional[str] = None,
        essence_type: Literal["vector", "matrix"] = "vector",
        n_essence_rows: int = 64,
        alpha_schedule: Optional[list[float]] = None,
        n_inner_layers: int = 0,
        use_exit_router: bool = False,
        exit_threshold: float = 0.5,
        exit_capacity_fraction: float = 0.5,
        use_recursive_kv_cache: bool = False,
        depth_families: Optional[int] = None,
        family_steps: Optional[list] = None,
        hypernetwork_attributes: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_steps = n_steps

        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )
        if essence_type == "matrix":
            self.essence = EssenceCoreMatrix(n_rows=n_essence_rows, d_model=d_model)
        else:
            self.essence = EssenceCore(
                d_essence,
                n_slots=1,
                init=essence_init,
                path=essence_path,
            )
        exit_router = None
        if use_exit_router:
            exit_router = ExitRouter(
                d_model=d_model,
                threshold=exit_threshold,
                capacity_fraction=exit_capacity_fraction,
            )
        self.tajalli_stack = TajalliStack(
            d_model=d_model,
            d_essence=d_model if essence_type == "matrix" else d_essence,
            n_heads=n_heads,
            d_head=d_head,
            d_ff=d_ff,
            n_steps=n_steps,
            dropout=dropout,
            max_seq_len=max_seq_len,
            essence_type=essence_type,
            n_essence_rows=n_essence_rows,
            alpha_schedule=alpha_schedule,
            n_inner_layers=n_inner_layers,
            exit_router=exit_router,
            use_recursive_kv_cache=use_recursive_kv_cache,
            depth_families=depth_families,
            family_steps=family_steps,
            hypernetwork_attributes=hypernetwork_attributes,
        )

    @property
    def output_head(self) -> nn.Linear:
        """Output head weight-tied with embeddings."""
        return nn.Linear(
            self.d_model,
            self.vocab_size,
            bias=False,
        )

    def get_output_weight(self) -> nn.Parameter:
        """Return shared weight for output (tie with embedding)."""
        return self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_step_metrics: bool = False,
        n_steps: Optional[int] = None,
        return_hidden: bool = False,
        memory_h: Optional[torch.Tensor] = None,
        mem_pos_start: int = 0,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        """
        Args:
            input_ids: (B, seq_len)
            mask: optional attention mask
            n_steps: if set, use this many recursive steps (else self.n_steps)
            return_hidden: if True, return hidden states (for adaptive softmax loss)
            memory_h: optional (B, mem_len, d_model) from previous segment for eval
            mem_pos_start: position offset of memory in full sequence
        Returns:
            logits or h: (B, seq_len, vocab_size) or (B, seq_len, d_model)
            step_metrics: if return_step_metrics
        """
        B = input_ids.shape[0]
        h = self.embedding(input_ids)
        essence = self.essence(B)
        h, step_metrics = self.tajalli_stack(
            h, essence, mask,
            return_step_metrics=return_step_metrics,
            n_steps_override=n_steps,
            memory_h=memory_h,
            mem_pos_start=mem_pos_start,
        )
        if return_hidden:
            return h, step_metrics
        logits = torch.matmul(h, self.embedding.weight.t())
        return logits, step_metrics


class TajalliModelPhase2(nn.Module):
    """
    Phase 2 model: same as Phase 1 but TajalliStack uses PairedMoELayer (use_moe=True).
    Weight-tied output head. Load Phase 1 checkpoint with strict=False and freeze essence.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_essence: int,
        n_heads: int,
        d_head: int,
        d_ff: int,
        n_steps: int = 6,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        padding_idx: Optional[int] = None,
        essence_init: str = "spectral",
        essence_path: Optional[str] = None,
        ortho_subset_frac: float = 0.1,
        expert_choice_routing: bool = False,
        n_experts: int = N_EXPERTS,
        n_attributes: Optional[int] = None,
        d_attr_hidden: Optional[int] = None,
        nonlinear_gate: bool = False,
        essence_type: Literal["vector", "matrix"] = "vector",
        n_essence_rows: int = 64,
        alpha_schedule: Optional[list[float]] = None,
        n_inner_layers: int = 0,
        use_exit_router: bool = False,
        exit_threshold: float = 0.5,
        exit_capacity_fraction: float = 0.5,
        use_recursive_kv_cache: bool = False,
        depth_families: Optional[int] = None,
        family_steps: Optional[list] = None,
        hypernetwork_attributes: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_steps = n_steps

        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )
        if essence_type == "matrix":
            self.essence = EssenceCoreMatrix(n_rows=n_essence_rows, d_model=d_model)
        else:
            self.essence = EssenceCore(
                d_essence,
                n_slots=1,
                init=essence_init,
                path=essence_path,
            )
        tajalli_kw = {}
        if n_attributes is not None:
            tajalli_kw["n_attributes"] = n_attributes
        if d_attr_hidden is not None:
            tajalli_kw["d_attr_hidden"] = d_attr_hidden
        tajalli_kw["nonlinear_gate"] = nonlinear_gate
        tajalli_kw["essence_type"] = essence_type
        tajalli_kw["n_essence_rows"] = n_essence_rows
        tajalli_kw["n_experts"] = n_experts
        tajalli_kw["alpha_schedule"] = alpha_schedule
        tajalli_kw["n_inner_layers"] = n_inner_layers
        tajalli_kw["use_recursive_kv_cache"] = use_recursive_kv_cache
        tajalli_kw["expert_choice_routing"] = expert_choice_routing
        tajalli_kw["depth_families"] = depth_families
        tajalli_kw["family_steps"] = family_steps
        tajalli_kw["hypernetwork_attributes"] = hypernetwork_attributes
        if use_exit_router:
            tajalli_kw["exit_router"] = ExitRouter(
                d_model=d_model,
                threshold=exit_threshold,
                capacity_fraction=exit_capacity_fraction,
            )
        self.tajalli_stack = TajalliStack(
            d_model=d_model,
            d_essence=d_model if essence_type == "matrix" else d_essence,
            n_heads=n_heads,
            d_head=d_head,
            d_ff=d_ff,
            n_steps=n_steps,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_moe=True,
            ortho_subset_frac=ortho_subset_frac,
            **tajalli_kw,
        )

    def get_output_weight(self) -> torch.nn.Parameter:
        return self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_step_metrics: bool = False,
        n_steps: Optional[int] = None,
        global_step: Optional[int] = None,
        rescue_experts: Optional[list] = None,
        return_hidden: bool = False,
        memory_h: Optional[torch.Tensor] = None,
        mem_pos_start: int = 0,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        # global_step, rescue_experts ignored (PairedMoE only; kept for API compatibility)
        B = input_ids.shape[0]
        h = self.embedding(input_ids)
        essence = self.essence(B)
        h, step_metrics = self.tajalli_stack(
            h, essence, mask,
            return_step_metrics=return_step_metrics,
            n_steps_override=n_steps,
            memory_h=memory_h,
            mem_pos_start=mem_pos_start,
        )
        if return_hidden:
            return h, step_metrics
        logits = torch.matmul(h, self.embedding.weight.t())
        return logits, step_metrics

    @classmethod
    def load_phase1_checkpoint(
        cls,
        phase1_ckpt_path: str,
        config: dict,
        vocab_size: Optional[int] = None,
        freeze_essence: bool = True,
    ) -> "TajalliModelPhase2":
        """
        Load Phase 1 (variable-depth) checkpoint into Phase 2 model. MoE keys are
        missing and stay randomly initialized. Optionally freeze essence.
        """
        path = Path(phase1_ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {phase1_ckpt_path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            state_dict = (
                ckpt.get("model_state_dict")
                or ckpt.get("state_dict")
                or ckpt.get("model")
            )
            if state_dict is None:
                state_dict = ckpt
        else:
            state_dict = ckpt

        vocab_size = vocab_size or config.get("vocab_size", 50257)
        tcfg = config.get("tajalli", {})
        ecfg = config.get("essence", {})
        essence_type = "matrix" if ecfg.get("type") == "matrix" else "vector"
        n_essence_rows = ecfg.get("n_rows", 64)
        d_essence = config.get("d_essence", config["d_model"])
        n_experts = config.get("n_experts", N_EXPERTS)
        model = cls(
            vocab_size=vocab_size,
            d_model=config["d_model"],
            d_essence=d_essence,
            n_heads=config["n_heads"],
            d_head=config["d_head"],
            d_ff=config["d_ff"],
            n_steps=config.get("recursive_steps", 6),
            max_seq_len=config.get("max_seq_len", 512),
            dropout=0.0,
            essence_init=config.get("essence_init", "spectral"),
            essence_path=config.get("essence_path"),
            ortho_subset_frac=config.get("ortho_subset_fraction", 0.1),
            expert_choice_routing=config.get("expert_choice_routing", False),
            n_experts=n_experts,
            n_attributes=tcfg.get("n_attributes"),
            d_attr_hidden=tcfg.get("d_attr_hidden"),
            nonlinear_gate=tcfg.get("nonlinear_gate", False),
            essence_type=essence_type,
            n_essence_rows=n_essence_rows,
            alpha_schedule=tcfg.get("alpha_schedule"),
            n_inner_layers=config.get("n_inner_layers", 0),
            use_exit_router=config.get("use_exit_router", False),
            exit_threshold=config.get("exit_threshold", 0.5),
            exit_capacity_fraction=config.get("exit_capacity_fraction", 0.5),
            use_recursive_kv_cache=config.get("use_recursive_kv_cache", False),
            depth_families=tcfg.get("depth_families"),
            family_steps=tcfg.get("family_steps"),
            hypernetwork_attributes=tcfg.get("hypernetwork_attributes", False),
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # Weight loading report: print ALL missing, unexpected, shape mismatches
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        common = model_keys & ckpt_keys
        shape_mismatches = [
            (k, state_dict[k].shape, model.state_dict()[k].shape)
            for k in common
            if state_dict[k].shape != model.state_dict()[k].shape
        ]
        print("\n" + "=" * 70)
        print("WEIGHT LOADING REPORT (Phase 1 -> Phase 2)")
        print("=" * 70)
        print(f"MISSING keys ({len(missing)} total, in Phase 2 model but NOT in Phase 1 ckpt):")
        for k in sorted(missing):
            print(f"  {k} {tuple(model.state_dict()[k].shape)}")
        print(f"\nUNEXPECTED keys ({len(unexpected)} total, in Phase 1 ckpt but NOT in Phase 2 model):")
        for k in sorted(unexpected):
            print(f"  {k} {tuple(state_dict[k].shape)}")
        if shape_mismatches:
            print(f"\nSHAPE MISMATCHES ({len(shape_mismatches)} keys in both, different shapes):")
            for k, c_shape, m_shape in shape_mismatches:
                print(f"  {k}: ckpt {tuple(c_shape)} vs model {tuple(m_shape)}")
        else:
            print("\nNo shape mismatches.")
        print("=" * 70 + "\n")

        # Explicitly set Phase 2 alphas from Phase 1 so Phase 2 starts with learned blending schedule
        alpha_key = "tajalli_stack.block.tajalli_layer.alpha_per_step"
        if alpha_key in state_dict:
            ckpt_alpha = state_dict[alpha_key]
            model_alpha = model.tajalli_stack.block.tajalli_layer.alpha_per_step
            if ckpt_alpha.shape == model_alpha.shape:
                with torch.no_grad():
                    model_alpha.copy_(ckpt_alpha)
                alphas_sigmoid = torch.sigmoid(ckpt_alpha).tolist()
                n_show = min(12, len(alphas_sigmoid))
                profile = [round(alphas_sigmoid[i], 3) for i in range(n_show)]
                print(f"Loaded alpha_per_step from Phase 1 (step 0→{n_show - 1}): {profile}")
            else:
                print(f"WARNING: alpha shape mismatch — ckpt {ckpt_alpha.shape} vs model {model_alpha.shape}; alphas not copied")

        # MoE expert initialization: copy Phase 1 FFN into all n_experts (PairedMoELayer)
        ffn_w1_key = "tajalli_stack.block.ffn.w1.weight"
        ffn_b1_key = "tajalli_stack.block.ffn.w1.bias"
        ffn_w2_key = "tajalli_stack.block.ffn.w2.weight"
        ffn_b2_key = "tajalli_stack.block.ffn.w2.bias"
        moe = model.tajalli_stack.block.moe_layer
        n_experts = moe.n_experts
        if all(k in state_dict for k in (ffn_w1_key, ffn_b1_key, ffn_w2_key, ffn_b2_key)):
            ffn_w1 = state_dict[ffn_w1_key]
            ffn_b1 = state_dict[ffn_b1_key]
            ffn_w2 = state_dict[ffn_w2_key]
            ffn_b2 = state_dict[ffn_b2_key]
            assert moe.w1.shape[1:] == ffn_w1.t().shape, (
                f"MoE w1 shape {moe.w1.shape[1:]} vs FFN w1.t() {ffn_w1.t().shape}"
            )
            assert moe.w2.shape[1:] == ffn_w2.t().shape, (
                f"MoE w2 shape {moe.w2.shape[1:]} vs FFN w2.t() {ffn_w2.t().shape}"
            )
            with torch.no_grad():
                for i in range(n_experts):
                    moe.w1[i].copy_(ffn_w1.t())
                    moe.b1[i].copy_(ffn_b1)
                    moe.w2[i].copy_(ffn_w2.t())
                    moe.b2[i].copy_(ffn_b2)
            print(f"Initialized {n_experts} MoE experts from Phase 1 FFN (d_ff={moe.d_ff})")

        if freeze_essence:
            for p in model.essence.parameters():
                p.requires_grad = False
        return model


class TajalliModelPhase3(nn.Module):
    """
    Phase 3: Phase 2 stack + Lawḥ (first/last step only, two-stage retrieval)
    + Barzakh bottleneck + optional Qadr. Load from Phase 2 checkpoint;
    Barzakh init is near-identity so loaded weights are not corrupted.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_essence: int,
        n_heads: int,
        d_head: int,
        d_ff: int,
        n_steps: int = 6,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        padding_idx: Optional[int] = None,
        essence_init: str = "spectral",
        essence_path: Optional[str] = None,
        ortho_subset_frac: float = 0.1,
        expert_choice_routing: bool = False,
        n_experts: int = N_EXPERTS,
        # Lawḥ
        lawh_store: Optional[LawhMemoryStore] = None,
        d_key: int = 384,
        d_value: int = 384,
        lawh_retrieval_k: int = 256,
        lawh_at_steps: Optional[list] = None,
        # Barzakh
        d_barzakh: int = 256,
        # Qadr (optional, usually eval-only)
        use_qadr: bool = False,
        qadr_repetition_penalty: float = 1.0,
        qadr_temperature: float = 1.0,
        # Jamba (replaces block attention + MoE when True)
        use_jamba: bool = False,
        jamba_config: Optional[dict] = None,
        # Tajallī layer (e.g. nonlinear heads when resuming from phase2_nonlinear_heads)
        n_attributes: Optional[int] = None,
        d_attr_hidden: Optional[int] = None,
        nonlinear_gate: bool = False,
        essence_type: Literal["vector", "matrix"] = "vector",
        n_essence_rows: int = 64,
        alpha_schedule: Optional[list[float]] = None,
        n_inner_layers: int = 0,
        use_exit_router: bool = False,
        exit_threshold: float = 0.5,
        exit_capacity_fraction: float = 0.5,
        use_recursive_kv_cache: bool = False,
        depth_families: Optional[int] = None,
        family_steps: Optional[list] = None,
        hypernetwork_attributes: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_steps = n_steps
        self.lawh_retrieval_k = lawh_retrieval_k
        self.use_qadr = use_qadr
        lawh_at_steps = lawh_at_steps or [0, -1]

        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )
        if essence_type == "matrix":
            self.essence = EssenceCoreMatrix(n_rows=n_essence_rows, d_model=d_model)
        else:
            self.essence = EssenceCore(
                d_essence,
                n_slots=1,
                init=essence_init,
                path=essence_path,
            )

        lawh_cross_attn = None
        if lawh_store is not None:
            lawh_cross_attn = LawhCrossAttention(
                d_model=d_model,
                d_key=d_key,
                d_value=d_value,
                dropout=dropout,
            )

        self.lawh_store = lawh_store
        tajalli_kw = {}
        if n_attributes is not None:
            tajalli_kw["n_attributes"] = n_attributes
        if d_attr_hidden is not None:
            tajalli_kw["d_attr_hidden"] = d_attr_hidden
        tajalli_kw["nonlinear_gate"] = nonlinear_gate
        tajalli_kw["essence_type"] = essence_type
        tajalli_kw["n_essence_rows"] = n_essence_rows
        tajalli_kw["n_experts"] = n_experts
        tajalli_kw["alpha_schedule"] = alpha_schedule
        tajalli_kw["n_inner_layers"] = n_inner_layers
        tajalli_kw["use_recursive_kv_cache"] = use_recursive_kv_cache
        tajalli_kw["expert_choice_routing"] = expert_choice_routing
        tajalli_kw["depth_families"] = depth_families
        tajalli_kw["family_steps"] = family_steps
        tajalli_kw["hypernetwork_attributes"] = hypernetwork_attributes
        if use_exit_router:
            tajalli_kw["exit_router"] = ExitRouter(
                d_model=d_model,
                threshold=exit_threshold,
                capacity_fraction=exit_capacity_fraction,
            )
        self.tajalli_stack = TajalliStack(
            d_model=d_model,
            d_essence=d_model if essence_type == "matrix" else d_essence,
            n_heads=n_heads,
            d_head=d_head,
            d_ff=d_ff,
            n_steps=n_steps,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_moe=True,
            ortho_subset_frac=ortho_subset_frac,
            lawh_cross_attn=lawh_cross_attn,
            lawh_at_steps=lawh_at_steps,
            use_jamba=use_jamba,
            jamba_config=jamba_config,
            **tajalli_kw,
        )
        self.barzakh = BarzakhBottleneck(d_model=d_model, d_barzakh=d_barzakh)
        self.final_norm = build_norm("layernorm", d_model)
        self.qadr = QadrConstraints(
            repetition_penalty=qadr_repetition_penalty,
            temperature=qadr_temperature,
            enable_repetition_penalty=use_qadr,
            enable_temperature=use_qadr,
        ) if use_qadr else None

    def get_output_weight(self) -> torch.nn.Parameter:
        return self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_step_metrics: bool = False,
        n_steps: Optional[int] = None,
        lawh_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        B = input_ids.shape[0]
        h = self.embedding(input_ids)
        essence = self.essence(B)

        if lawh_kv is None and self.lawh_store is not None:
            # Two-stage retrieval: one query per batch (mean over B, T), top-K for whole batch
            with torch.no_grad():
                query = h.mean(dim=(0, 1), keepdim=True)
                query = self.tajalli_stack.block.lawh_cross_attn.query_proj(query)
                indices = self.lawh_store.retrieve_topk(
                    query, k=self.lawh_retrieval_k, device=h.device
                )
                idx = indices.view(-1)
                if idx.dim() == 0:
                    idx = idx.unsqueeze(0)
                lawh_kv = self.lawh_store.get_keys_values(idx)
                lawh_kv = (lawh_kv[0].to(h.device), lawh_kv[1].to(h.device))

        h, step_metrics = self.tajalli_stack(
            h, essence, mask,
            return_step_metrics=return_step_metrics,
            n_steps_override=n_steps,
            lawh_kv=lawh_kv,
        )
        h_pre = h
        h = self.barzakh(h)
        # Output path in float32 so logits never overflow in fp16 (stack can produce large h)
        h = h.float()
        h = self.final_norm(h)
        if step_metrics is None:
            step_metrics = {}
        if self.training:
            step_metrics["barzakh_reconstruction_loss"] = torch.nn.functional.mse_loss(h_pre.float(), h)
        logits = torch.matmul(h, self.embedding.weight.t().float())
        if self.qadr is not None:
            logits = self.qadr(logits, input_ids=input_ids)
        return logits, step_metrics

    @classmethod
    def load_phase2_checkpoint(
        cls,
        phase2_ckpt_path: str,
        config: dict,
        lawh_store_path: Optional[str] = None,
        vocab_size: Optional[int] = None,
        freeze_essence: bool = True,
        skip_phase2_weights: bool = False,
    ) -> "TajalliModelPhase3":
        """
        Load Phase 2 checkpoint into Phase 3 model. New components (lawh_cross_attn,
        barzakh) are initialized; Barzakh decode is near-identity so loss does not spike.
        When skip_phase2_weights=True, only builds the model structure (no Phase 2 load);
        use when loading a Phase 3 checkpoint with differing architecture (e.g. d_attr_hidden).
        """
        path = Path(phase2_ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {phase2_ckpt_path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            state_dict = (
                ckpt.get("model_state_dict")
                or ckpt.get("state_dict")
                or ckpt.get("model")
            )
            if state_dict is None:
                state_dict = ckpt
        else:
            state_dict = ckpt

        vocab_size = vocab_size or config.get("vocab_size", 50257)
        lawh_store = None
        if lawh_store_path and Path(lawh_store_path).exists():
            lawh_store = LawhMemoryStore.load_from_file(lawh_store_path)

        d_key = config.get("d_key", 384)
        d_value = config.get("d_value", 384)
        if lawh_store is not None:
            d_key = lawh_store.d_key
            d_value = lawh_store.d_value

        tcfg = config.get("tajalli", {})
        ecfg = config.get("essence", {})
        essence_type = "matrix" if ecfg.get("type") == "matrix" else "vector"
        n_essence_rows = ecfg.get("n_rows", 64)
        d_essence = config.get("d_essence", config["d_model"])
        model = cls(
            vocab_size=vocab_size,
            d_model=config["d_model"],
            d_essence=d_essence,
            n_heads=config["n_heads"],
            d_head=config["d_head"],
            d_ff=config["d_ff"],
            n_steps=config.get("recursive_steps", 6),
            max_seq_len=config.get("max_seq_len", 512),
            dropout=0.0,
            essence_init=config.get("essence_init", "spectral"),
            essence_path=config.get("essence_path"),
            ortho_subset_frac=config.get("ortho_subset_fraction", 0.1),
            expert_choice_routing=config.get("expert_choice_routing", False),
            n_experts=config.get("n_experts", N_EXPERTS),
            alpha_schedule=tcfg.get("alpha_schedule"),
            lawh_store=lawh_store,
            d_key=d_key,
            d_value=d_value,
            lawh_retrieval_k=config.get("lawh_retrieval_k", 256),
            lawh_at_steps=config.get("lawh_at_steps", [0, -1]),
            d_barzakh=config.get("d_barzakh", 256),
            use_qadr=config.get("use_qadr", False),
            use_jamba=config.get("use_jamba", False),
            jamba_config=config.get("jamba_config"),
            n_attributes=tcfg.get("n_attributes"),
            d_attr_hidden=tcfg.get("d_attr_hidden"),
            nonlinear_gate=tcfg.get("nonlinear_gate", False),
            essence_type=essence_type,
            n_essence_rows=n_essence_rows,
            n_inner_layers=config.get("n_inner_layers", 0),
            use_exit_router=config.get("use_exit_router", False),
            exit_threshold=config.get("exit_threshold", 0.5),
            exit_capacity_fraction=config.get("exit_capacity_fraction", 0.5),
            use_recursive_kv_cache=config.get("use_recursive_kv_cache", False),
            depth_families=tcfg.get("depth_families"),
            family_steps=tcfg.get("family_steps"),
            hypernetwork_attributes=tcfg.get("hypernetwork_attributes", False),
        )
        if not skip_phase2_weights:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        else:
            missing, unexpected = [], []

        # Explicitly set Phase 3 alphas from Phase 2 so Phase 3 keeps the learned blending schedule
        alpha_key = "tajalli_stack.block.tajalli_layer.alpha_per_step"
        if alpha_key in state_dict and not skip_phase2_weights:
            ckpt_alpha = state_dict[alpha_key]
            layer = getattr(getattr(model.tajalli_stack, "block", None), "tajalli_layer", None)
            if layer is not None and hasattr(layer, "alpha_per_step"):
                model_alpha = layer.alpha_per_step
                if ckpt_alpha.shape == model_alpha.shape:
                    with torch.no_grad():
                        model_alpha.copy_(ckpt_alpha)
                    alphas_sigmoid = torch.sigmoid(ckpt_alpha).tolist()
                    n_show = min(12, len(alphas_sigmoid))
                    profile = [round(alphas_sigmoid[i], 3) for i in range(n_show)]
                    print(f"Loaded alpha_per_step from Phase 2 (step 0→{n_show - 1}): {profile}")

        # When model has Jamba, copy old block.attention and block.moe_layer into jamba_block.attn / moe_layer
        block = getattr(model.tajalli_stack, "block", None)
        if block is not None and getattr(block, "jamba_block", None) is not None:
            remap = {}
            prefix_old = "tajalli_stack.block."
            prefix_new = "tajalli_stack.block.jamba_block."
            for key, value in state_dict.items():
                if key.startswith(prefix_old + "attention."):
                    remap[prefix_new + "attn." + key[len(prefix_old + "attention."):]] = value
                elif key.startswith(prefix_old + "moe_layer."):
                    remap[prefix_new + "moe_layer." + key[len(prefix_old + "moe_layer."):]] = value
                elif key.startswith(prefix_old + "norm_attn."):
                    remap[prefix_new + "norm_attn." + key[len(prefix_old + "norm_attn."):]] = value
                elif key.startswith(prefix_old + "norm_ffn."):
                    remap[prefix_new + "norm_ffn." + key[len(prefix_old + "norm_ffn."):]] = value
            if remap:
                model.load_state_dict(remap, strict=False)
                print(f"Remapped {len(remap)} keys from old attention/MoE into Jamba block (SSM left randomly initialized).")

        if missing:
            for k in list(missing)[:10]:
                if "lawh" in k or "barzakh" in k:
                    continue
            # Expected missing: lawh_cross_attn.*, barzakh.*, jamba_block.ssm.* when loading old ckpt
        if freeze_essence:
            for p in model.essence.parameters():
                p.requires_grad = False
        return model
