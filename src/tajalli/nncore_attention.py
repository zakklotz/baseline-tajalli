from __future__ import annotations

import inspect
from typing import Any, Optional

import torch
import torch.nn as nn

from tajalli.nncore_bridge import _require_nncore


class TajalliAttention(nn.Module):
    """
    nn-core backed attention adapter.

    Keeps Tajalli's legacy signature:
        forward(x, mask=None, *, memory_h=None, cached_kv=None, mem_pos_start=0, return_kv=False)

    IMPORTANT BEHAVIOR (to match tajalli_deprecated Phase1):
      - RoPE must be applied to projected q/k (not to x before projection).
      - Use PyTorch SDPA backend (nncore backend="sdpa") for stable loss scale.
      - Treat Tajalli's (B, T) mask as KEY PADDING MASK, not an attention matrix.
        Tajalli convention: mask is 1=keep, 0=pad.
        nn-core convention: key_padding_mask is bool with True=keep, False=mask.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        # Kept for legacy call sites; nn-core derives head dim from d_model//n_heads.
        d_head: Optional[int] = None,  # noqa: ARG002
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,  # noqa: ARG002 (nn-core Rope uses fixed base; kept for API)
        bias: bool = True,
        causal: bool = True,
        use_kv_cache: bool = False,
    ):
        super().__init__()
        _require_nncore()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        from nncore.layers.attention import MultiheadAttention  # type: ignore

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.causal = bool(causal)

        # Build nn-core attention with "sdpa" + "rope" to match deprecated behavior.
        ctor = inspect.signature(MultiheadAttention.__init__)
        kwargs: dict[str, Any] = {}

        # Required dims / heads (support minor name variations if any)
        if "d_model" in ctor.parameters:
            kwargs["d_model"] = d_model
        elif "embed_dim" in ctor.parameters:
            kwargs["embed_dim"] = d_model
        elif "dim" in ctor.parameters:
            kwargs["dim"] = d_model
        else:
            # Very unlikely for nn-core, but keep a clear message.
            raise TypeError("nn-core MultiheadAttention ctor missing d_model/embed_dim/dim")

        if "num_heads" in ctor.parameters:
            kwargs["num_heads"] = n_heads
        elif "n_heads" in ctor.parameters:
            kwargs["n_heads"] = n_heads
        elif "heads" in ctor.parameters:
            kwargs["heads"] = n_heads
        else:
            raise TypeError("nn-core MultiheadAttention ctor missing num_heads/n_heads/heads")

        if "bias" in ctor.parameters:
            kwargs["bias"] = bool(bias)

        # nn-core uses these exact names
        if "attn_dropout_p" in ctor.parameters:
            kwargs["attn_dropout_p"] = float(dropout)
        if "out_dropout_p" in ctor.parameters:
            kwargs["out_dropout_p"] = float(dropout)

        # Force stable backend + correct RoPE placement
        if "backend" in ctor.parameters:
            kwargs["backend"] = "sdpa"
        if "positional" in ctor.parameters:
            kwargs["positional"] = "rope"
        if "max_seq_len" in ctor.parameters:
            kwargs["max_seq_len"] = int(max_seq_len)

        if "use_kv_cache" in ctor.parameters:
            kwargs["use_kv_cache"] = bool(use_kv_cache)

        self.attn = MultiheadAttention(**kwargs)

    def _split_mask(
        self, mask: Optional[torch.Tensor], *, T: int, device: torch.device
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns (key_padding_mask, attn_mask)

        - If mask is (B, T): treat as key_padding_mask (True=keep).
        - If mask is (T, T) or (B, T, T) or (B, 1, T, T) etc: treat as attn_mask.
        """
        if mask is None:
            return None, None

        m = mask.to(device)

        # (B, T) pad mask
        if m.dim() == 2 and m.shape[1] == T:
            if m.dtype != torch.bool:
                m = m.to(torch.bool)
            return m, None

        # Otherwise, treat as attn mask (bool keep-mask or additive)
        # nn-core expects broadcastable to (B,H,Tq,Tk) or (Tq,Tk)
        return None, m

    def _call_attn(
        self,
        x: torch.Tensor,
        *,
        context: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        pos_offset: int,
    ) -> Any:
        """
        Call nn-core attention with best-effort signature adaptation.
        (Your nn-core.zip uses: forward(x, context=..., attn_mask=..., key_padding_mask=..., is_causal=..., pos_offset=...))
        """
        fwd = self.attn.forward
        sig = inspect.signature(fwd)

        def filtered_kwargs(**kw: Any) -> dict[str, Any]:
            return {k: v for k, v in kw.items() if (k in sig.parameters and v is not None)}

        kw: dict[str, Any] = {}
        if context is not None:
            if "context" in sig.parameters:
                kw["context"] = context
            elif "kv" in sig.parameters:
                kw["kv"] = context
            elif "key_value" in sig.parameters:
                kw["key_value"] = context

        kw.update(filtered_kwargs(attn_mask=attn_mask))
        kw.update(filtered_kwargs(key_padding_mask=key_padding_mask))

        # causal flag name can vary
        if "is_causal" in sig.parameters:
            kw["is_causal"] = is_causal
        elif "causal" in sig.parameters:
            kw["causal"] = is_causal

        # position offset for RoPE continuity
        if "pos_offset" in sig.parameters:
            kw["pos_offset"] = int(pos_offset)

        return self.attn(x, **kw)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        memory_h: Optional[torch.Tensor] = None,
        cached_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # noqa: ARG002 (legacy)
        mem_pos_start: int = 0,
        return_kv: bool = False,
    ):
        """
        Returns either:
          - y (B,T,C)
          - or (y, kv) if return_kv=True and backend supports it (best-effort)
        """
        B, T, C = x.shape
        _ = B
        _ = C

        key_padding_mask, attn_mask = self._split_mask(mask, T=T, device=x.device)

        # Only use memory_h as context (Phase1 usually None)
        context = memory_h if memory_h is not None else None

        out = self._call_attn(
            x,
            context=context,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            is_causal=self.causal,
            pos_offset=int(mem_pos_start),
        )

        # Normalize return types
        if isinstance(out, tuple):
            y = out[0]
        else:
            y = out

        if isinstance(y, dict):
            if "out" in y:
                y = y["out"]
            elif "y" in y:
                y = y["y"]

        if return_kv:
            if isinstance(out, tuple) and len(out) >= 2:
                return y, out[1]
            return y, None

        return y
