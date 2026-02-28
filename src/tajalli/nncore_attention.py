from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Any
from nncore.layers.attention import MultiheadAttention
import inspect


def _require_nncore():
    try:
        import nncore  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "nn-core is not installed.\n"
            "Run:\n"
            "  pip install -e ./.nn-core"
        ) from e


def _construct_mha(MultiheadAttention, *, d_model: int, n_heads: int, dropout: float, bias: bool):
    """
    Construct nn-core MultiheadAttention across API variants by only passing
    kwargs that exist in the constructor signature.
    """
    sig = inspect.signature(MultiheadAttention.__init__)
    params = sig.parameters

    kwargs = {}

    # d_model is consistent
    if "d_model" in params:
        kwargs["d_model"] = d_model
    elif "embed_dim" in params:
        kwargs["embed_dim"] = d_model

    # head count name varies
    if "num_heads" in params:
        kwargs["num_heads"] = n_heads
    elif "n_heads" in params:
        kwargs["n_heads"] = n_heads
    elif "heads" in params:
        kwargs["heads"] = n_heads

    # bias is common
    if "bias" in params:
        kwargs["bias"] = bias

    # dropout name varies or might not exist at all
    if "dropout" in params:
        kwargs["dropout"] = dropout
    elif "attn_dropout" in params:
        kwargs["attn_dropout"] = dropout
    elif "p" in params:
        kwargs["p"] = dropout

    return MultiheadAttention(**kwargs)


def _build_rope(head_dim: int, rope_theta: float) -> Optional[Any]:
    """
    Best-effort construction of a RoPE/rotary embedding module from nn-core.

    nn-core API may vary across revisions. We attempt multiple class names/paths.
    If none are found, return None (attention will run without RoPE).
    """
    _require_nncore()

    # Try common exports / names (most -> least likely).
    candidates = [
        ("nncore.positional", "RotaryEmbedding"),
        ("nncore.positional", "RotaryPositionalEmbedding"),
        ("nncore.positional", "Rope"),
        ("nncore.positional", "RoPE"),
        ("nncore.positional.rope", "RotaryEmbedding"),
        ("nncore.positional.rope", "Rope"),
    ]

    for mod_name, cls_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            # Try common ctor patterns
            try:
                return cls(dim=head_dim, base=rope_theta)
            except TypeError:
                try:
                    return cls(head_dim=head_dim, theta=rope_theta)
                except TypeError:
                    try:
                        return cls(dim=head_dim, theta=rope_theta)
                    except TypeError:
                        try:
                            return cls(head_dim, rope_theta)
                        except TypeError:
                            continue
        except Exception:
            continue

    return None


class TajalliAttention(nn.Module):
    """
    Drop-in replacement for Tajalli's previous MultiHeadAttention.

    Internally delegates to nn-core MultiheadAttention.
    RoPE is applied if we can construct a compatible rotary module from nn-core.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        bias: bool = True,
        causal: bool = True,
    ):
        super().__init__()
        _require_nncore()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")


        self.d_model = d_model
        self.n_heads = n_heads
        self.causal = causal

        head_dim = d_model // n_heads
        self.rope = _build_rope(head_dim=head_dim, rope_theta=rope_theta)

        self.attn = _construct_mha(
            MultiheadAttention,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
        )

    def _call_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        nn-core attention signature may vary. Try a few compatible call styles.
        """
        # 1) Most explicit: query/key/value + rope + is_causal
        if self.rope is not None:
            try:
                return self.attn(
                    q, k, v,
                    rope=self.rope,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                )
            except TypeError:
                pass

        # 2) Same but without rope kw
        try:
            return self.attn(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
        except TypeError:
            pass

        # 3) Some implementations may accept (x) for self-attn
        if q is k and k is v:
            # try with causal first
            try:
                return self.attn(q, attn_mask=attn_mask, is_causal=is_causal)
            except TypeError:
                pass
            try:
                return self.attn(q, attn_mask=attn_mask)
            except TypeError:
                pass

        # 4) Last resort: just call positional args
        return self.attn(q, k, v)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, T, C)
        context: optional cross-attention source (B, S, C)
        """
        if context is None:
            return self._call_attn(
                x, x, x,
                is_causal=self.causal,
                attn_mask=attn_mask,
            )

        # Cross-attention: causal should be False
        return self._call_attn(
            x, context, context,
            is_causal=False,
            attn_mask=attn_mask,
        )
