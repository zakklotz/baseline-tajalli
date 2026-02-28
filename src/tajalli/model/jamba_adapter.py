"""
Jamba-style block: SSM + Attention + Asmā' MoE. Used directly in TajalliBlockV2 when use_jamba=True.
Stateless per forward (SSM state reset each call). No gate; block replaces self-attention and MoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .attention import MultiHeadAttention


class SimpleSSMLayer(nn.Module):
    """
    Minimal discrete-time SSM layer (stateless per forward): conv1d + linear recurrence.
    Matches Jamba-tiny hyperparams: d_state=16, dt_rank=32, expand=2. No selective scan;
    uses learned A, B, C for compatibility. State reset each forward.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = min(dt_rank or (d_model // 16), self.d_inner)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.proj_in = nn.Linear(d_model, self.d_inner * 2)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.log(0.99 - torch.rand(self.d_inner, d_state) * 0.01))
        self.B = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.1)
        self.proj_out = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        xz = self.proj_in(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv(x_conv)[:, :, :T]
        x_conv = x_conv.transpose(1, 2)
        u = F.silu(x_conv)
        dt_in = x_in[:, :, : self.dt_rank] if x_in.shape[-1] >= self.dt_rank else x_in
        if dt_in.shape[-1] < self.dt_rank:
            dt_in = F.pad(dt_in, (0, self.dt_rank - dt_in.shape[-1]))
        dt = F.softplus(self.dt_proj(dt_in))
        A = -torch.exp(self.A_log)
        out = self._scan(u, dt, A)
        out = out * F.silu(z)
        out = self.proj_out(out) + x * self.D.mean()
        return out

    def _scan(self, u: torch.Tensor, dt: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, T, D = u.shape
        N = A.shape[1]
        A = A.to(u.device)
        B_mat = self.B.to(u.device)
        C_mat = self.C.to(u.device)
        h = torch.zeros(B, D, N, device=u.device, dtype=u.dtype)
        outs = []
        for t in range(T):
            dt_t = dt[:, t, :]
            u_t = u[:, t, :]
            h = h * torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0)) + dt_t.unsqueeze(-1) * B_mat.unsqueeze(0) * u_t.unsqueeze(-1)
            y_t = (h * C_mat.unsqueeze(0)).sum(dim=-1)
            outs.append(y_t)
        return torch.stack(outs, dim=1)


class JambaStyleBlock(nn.Module):
    """
    Jamba-style block: SSM -> residual+norm -> Attention -> residual+norm -> Asmā' MoE -> residual+norm.
    Replaces self-attention and standalone MoE in the recursive step. PairedMoELayer (14 experts, 7 pairs).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_head: Optional[int] = None,
        d_ff: int = 2048,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        ortho_subset_frac: float = 0.1,
        mamba_d_state: int = 16,
        mamba_dt_rank: int = 32,
        mamba_expand: int = 2,
        mamba_d_conv: int = 4,
    ):
        super().__init__()
        from .moe import PairedMoELayer
        d_head = d_head or (d_model // n_heads)
        self.norm_ssm = nn.LayerNorm(d_model)
        self.ssm = SimpleSSMLayer(
            d_model,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dt_rank=mamba_dt_rank,
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, d_head, dropout, max_seq_len)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.moe_layer = PairedMoELayer(
            d_model,
            d_ff=d_ff,
            top_k=2,
            ortho_subset_frac=ortho_subset_frac,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict]]:
        x = self.norm_ssm(x + self.ssm(x))
        x = self.norm_attn(x + self.attn(x, mask))
        moe_out, moe_aux = self.moe_layer(x, mask=mask, return_aux=return_aux)
        x = self.norm_ffn(x + moe_out)
        if return_aux:
            return x, {"moe_aux": moe_aux}
        return x, None
