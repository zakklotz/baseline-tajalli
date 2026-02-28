from __future__ import annotations

import inspect
from typing import Optional, Any

import torch
import torch.nn as nn


def _require_nncore():
    try:
        import nncore  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "nn-core is not installed.\n"
            "Run:\n"
            "  pip install -e ./.nn-core"
        ) from e


def _import_first(paths: list[tuple[str, str]]) -> Any:
    """
    Try importing (module, symbol) pairs in order. Return the symbol.
    Raise ImportError if none are found.
    """
    last_err: Optional[Exception] = None
    for mod_name, sym in paths:
        try:
            mod = __import__(mod_name, fromlist=[sym])
            return getattr(mod, sym)
        except Exception as e:
            last_err = e
            continue
    raise ImportError(f"Could not import any of: {paths}") from last_err


def build_norm(norm_type: str, d_model: int, eps: float = 1e-5) -> nn.Module:
    """
    Build a normalization module using nn-core if available.

    norm_type supports: 'rmsnorm', 'layernorm' (case-insensitive)
    """
    _require_nncore()
    nt = norm_type.lower()

    # Prefer nn-core norms; fall back to torch LayerNorm if needed.
    if nt in ("rms", "rmsnorm"):
        # Try common nn-core RMSNorm locations/names.
        RMSNorm = _import_first([
            ("nncore.layers.norm", "RMSNorm"),
            ("nncore.layers.norms", "RMSNorm"),
            ("nncore.layers.normalization", "RMSNorm"),
            ("nncore.norm", "RMSNorm"),
        ])
        # Constructor signatures vary: RMSNorm(d, eps=...) or RMSNorm(dim=..., eps=...)
        try:
            return RMSNorm(d_model, eps=eps)
        except TypeError:
            return RMSNorm(dim=d_model, eps=eps)

    if nt in ("ln", "layernorm", "layer_norm"):
        # Prefer nn-core LN wrapper if present, else torch
        try:
            LayerNorm = _import_first([
                ("nncore.layers.norm", "LayerNorm"),
                ("nncore.layers.norms", "LayerNorm"),
                ("nncore.layers.normalization", "LayerNorm"),
                ("nncore.norm", "LayerNorm"),
            ])
            try:
                return LayerNorm(d_model, eps=eps)
            except TypeError:
                return LayerNorm(normalized_shape=d_model, eps=eps)
        except ImportError:
            return nn.LayerNorm(d_model, eps=eps)

    raise ValueError(f"Unknown norm_type: {norm_type}")


def _construct_ffn(FFNCls: Any, *, d_model: int, d_ff: int, dropout: float, bias: bool, activation: str) -> nn.Module:
    """
    Construct nn-core MLP/FFN across API variants by inspecting the ctor signature
    and only passing supported kwargs.
    """
    sig = inspect.signature(FFNCls.__init__)
    params = sig.parameters
    kwargs = {}

    # dims
    if "d_model" in params:
        kwargs["d_model"] = d_model
    elif "embed_dim" in params:
        kwargs["embed_dim"] = d_model
    elif "dim" in params:
        kwargs["dim"] = d_model

    if "d_ff" in params:
        kwargs["d_ff"] = d_ff
    elif "hidden_dim" in params:
        kwargs["hidden_dim"] = d_ff
    elif "d_hidden" in params:
        kwargs["d_hidden"] = d_ff
    elif "mlp_dim" in params:
        kwargs["mlp_dim"] = d_ff

    # dropout (might not exist)
    if "dropout" in params:
        kwargs["dropout"] = dropout
    elif "p" in params:
        kwargs["p"] = dropout
    elif "mlp_dropout" in params:
        kwargs["mlp_dropout"] = dropout

    # bias
    if "bias" in params:
        kwargs["bias"] = bias

    # activation naming differs
    if "activation" in params:
        kwargs["activation"] = activation
    elif "act" in params:
        kwargs["act"] = activation

    return FFNCls(**kwargs)


class TajalliFFN(nn.Module):
    """
    Drop-in feed-forward network wrapper for Tajalli blocks.

    Internally uses an nn-core MLP/FFN implementation when available, with a
    torch fallback only if absolutely necessary.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "silu",
        bias: bool = True,
    ):
        super().__init__()
        _require_nncore()

        # Try to find a suitable nn-core FFN/MLP
        FFNCls = _import_first([
            ("nncore.layers.mlp", "MLP"),
            ("nncore.layers.ffn", "FeedForward"),
            ("nncore.layers.mlp", "FeedForward"),
            ("nncore.layers.transformer", "FeedForward"),
        ])

        try:
            self.ffn = _construct_ffn(
                FFNCls,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                bias=bias,
                activation=activation,
            )
        except Exception:
            # Last resort fallback (should rarely happen)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=bias),
                nn.SiLU() if activation.lower() in ("silu", "swish") else nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model, bias=bias),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
