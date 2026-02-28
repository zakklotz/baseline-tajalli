"""Multi-head attention with RoPE and feed-forward network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequency coefficients."""
    freqs = 1.0 / (theta ** (torch.arange(dim, dtype=torch.float32)[::2] / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_offset: int = 0,
    k_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K. Offsets for segment memory."""
    seq_len_q, seq_len_k = q.shape[2], k.shape[2]
    d_head = q.shape[-1]
    freqs_q = freqs_cis[q_offset : q_offset + seq_len_q].to(q.device)
    freqs_k = freqs_cis[k_offset : k_offset + seq_len_k].to(k.device)
    freqs_q = freqs_q.unsqueeze(0).unsqueeze(0)
    freqs_k = freqs_k.unsqueeze(0).unsqueeze(0)
    q_pairs = q.float().reshape(*q.shape[:-1], d_head // 2, 2)
    k_pairs = k.float().reshape(*k.shape[:-1], d_head // 2, 2)
    q_complex = torch.view_as_complex(q_pairs)
    k_complex = torch.view_as_complex(k_pairs)
    q_rot = torch.view_as_real(q_complex * freqs_q).flatten(-2)
    k_rot = torch.view_as_real(k_complex * freqs_k).flatten(-2)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head or (d_model // n_heads)
        self.dropout = dropout

        self.wq = nn.Linear(d_model, n_heads * self.d_head)
        self.wk = nn.Linear(d_model, n_heads * self.d_head)
        self.wv = nn.Linear(d_model, n_heads * self.d_head)
        self.wo = nn.Linear(n_heads * self.d_head, d_model)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.d_head, max_seq_len),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        memory_h: Optional[torch.Tensor] = None,
        mem_pos_start: int = 0,
        cached_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        When cached_kv is provided (K_0, V_0 from step 0), only Q is computed from x (MoR-style KV sharing).
        When return_kv is True and cached_kv is None, return (out, (K, V)) for the caller to cache.
        """
        B, T, C = x.shape

        if cached_kv is not None:
            k_cached, v_cached = cached_kv
            q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            q_rot, _ = apply_rope(q, k_cached, self.freqs_cis)
            dropout_p = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q_rot, k_cached, v_cached,
                dropout_p=dropout_p,
                is_causal=True,
            )
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.wo(out)

        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k_cur = self.wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v_cur = self.wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if memory_h is not None:
            mem_len = memory_h.shape[1]
            k_mem = self.wk(memory_h).view(B, mem_len, self.n_heads, self.d_head).transpose(1, 2)
            v_mem = self.wv(memory_h).view(B, mem_len, self.n_heads, self.d_head).transpose(1, 2)
            q, k_cur = apply_rope(q, k_cur, self.freqs_cis, q_offset=mem_pos_start + mem_len, k_offset=mem_pos_start + mem_len)
            _, k_mem = apply_rope(k_mem, k_mem, self.freqs_cis, q_offset=mem_pos_start, k_offset=mem_pos_start)
            k = torch.cat([k_mem, k_cur], dim=2)
            v = torch.cat([v_mem, v_cur], dim=2)
            # Causal: current can attend to all memory; causal within current
            mem_mask = torch.ones(T, mem_len, device=x.device, dtype=torch.bool)
            causal_cur = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            attn_mask = torch.cat([mem_mask, causal_cur], dim=1)
            dropout_p = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
                dropout_p=dropout_p, scale=1.0 / (self.d_head ** 0.5),
            )
        else:
            q, k_cur = apply_rope(q, k_cur, self.freqs_cis)
            dropout_p = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k_cur, v_cur,
                dropout_p=dropout_p,
                is_causal=True,
            )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        if return_kv:
            return self.wo(out), (k_cur, v_cur)
        return self.wo(out)


class FeedForward(nn.Module):
    """Standard FFN: Linear -> GELU -> Linear."""

    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or (4 * d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(
            self.w2(F.gelu(self.w1(x))),
            p=self.dropout,
            training=self.training,
        )
