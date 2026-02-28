"""
TajalliBlockV2: when use_jamba=True, Jamba block REPLACES self-attention and Asmā' MoE.
Order: Tajallī Regeneration -> Jamba Block (SSM + Attention + MoE) -> Lawḥ -> Tajallī blend.
No gate. No separate attention or MoE when Jamba is active.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, TYPE_CHECKING

from .tajalli import TajalliLayer, MAX_TAJALLI_STEPS, N_ATTRIBUTES
from .attention import MultiHeadAttention, FeedForward
from .jamba_adapter import JambaStyleBlock

if TYPE_CHECKING:
    from .moe import PairedMoELayer
    from .lawh import LawhCrossAttention


class TajalliBlockV2(nn.Module):
    """
    When use_jamba=True: Tajallī -> Jamba block (SSM + Attn + MoE) -> Lawḥ -> blend.
    When use_jamba=False: same as TajalliBlock (Attn + optional MoE/FFN + Lawḥ).
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
        moe_layer: Optional["PairedMoELayer"] = None,
        lawh_cross_attn: Optional["LawhCrossAttention"] = None,
        jamba_block: Optional[JambaStyleBlock] = None,
        use_jamba: bool = True,
        n_attributes: int = N_ATTRIBUTES,
        d_attr_hidden: Optional[int] = None,
        nonlinear_gate: bool = False,
        essence_type: Literal["vector", "matrix"] = "vector",
        n_essence_rows: int = 64,
        alpha_schedule: Optional[list[float]] = None,
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
        )
        self.jamba_block = jamba_block
        self.use_jamba = use_jamba
        self.lawh_cross_attn = lawh_cross_attn

        if use_jamba:
            self.attention = None
            self.moe_layer = None
            self.ffn = None
            self.norm_attn = None
            self.norm_ffn = None
        else:
            self.attention = MultiHeadAttention(
                d_model, n_heads, d_head, dropout, max_seq_len
            )
            self.moe_layer = moe_layer
            self.ffn = FeedForward(d_model, d_ff, dropout) if moe_layer is None else None
            self.norm_attn = nn.LayerNorm(d_model)
            self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,
        essence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        lawh_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_lawh_this_step: bool = False,
        step_idx: int = 0,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        h_prev = h
        tajalli_signal, tajalli_metrics = self.tajalli_layer(essence, h_prev, step_idx=step_idx)

        if self.use_jamba and self.jamba_block is not None:
            f_h, jamba_aux = self.jamba_block(h_prev, mask=mask, return_aux=return_metrics)
        else:
            jamba_aux = None
            f_h = self.norm_attn(h_prev + self.attention(h_prev, mask))

        if use_lawh_this_step and self.lawh_cross_attn is not None and lawh_kv is not None:
            K_batch, V_batch = lawh_kv
            f_h, lawh_metrics = self.lawh_cross_attn(
                f_h, K_batch, V_batch, return_metrics=return_metrics
            )
        else:
            lawh_metrics = None

        if not (self.use_jamba and self.jamba_block is not None):
            if self.moe_layer is not None:
                moe_out, moe_aux = self.moe_layer(f_h, mask=mask, return_aux=return_metrics)
                f_h = self.norm_ffn(f_h + moe_out)
            else:
                moe_aux = None
                f_h = self.norm_ffn(f_h + self.ffn(f_h))
        else:
            moe_aux = jamba_aux.get("moe_aux") if jamba_aux else None

        step_idx_clamped = min(step_idx, MAX_TAJALLI_STEPS - 1)
        alpha = torch.sigmoid(self.tajalli_layer.alpha_per_step[step_idx_clamped])
        h_new = alpha * tajalli_signal + (1 - alpha) * f_h

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
            return h_new, metrics
        return h_new, None
