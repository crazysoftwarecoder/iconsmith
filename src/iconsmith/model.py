"""Set-conditioned icon transformer.

Decoder-only transformer over path tokens, conditioned on a prefix of
``[stroke-width bucket] [name embedding] [K exemplar style vectors]``.
Exemplars are encoded by a small bidirectional transformer that shares the token embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from iconsmith.data import K
from iconsmith.dataset_build import N_STROKE_BUCKETS
from iconsmith.tokenizer import PAD, VOCAB_SIZE


@dataclass
class ModelConfig:
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1408
    enc_d_model: int = 384
    enc_layers: int = 4
    enc_heads: int = 6
    max_len: int = 1024
    text_dim: int = 384
    k: int = K
    p_drop_style: float = 0.1
    p_drop_text: float = 0.1
    dropout: float = 0.0
    vocab_size: int = VOCAB_SIZE
    n_stroke_buckets: int = N_STROKE_BUCKETS


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


def rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)  # [S, D/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [B, H, S, D]; cos/sin: [S, D/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor, mask: Tensor | None, cos: Tensor, sin: Tensor) -> Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # mask=None means causal via the fused kernel (no materialised S x S mask).
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=mask is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(y.transpose(1, 2).reshape(b, s, d))


class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = Attention(d, n_heads, dropout)
        self.n2 = RMSNorm(d)
        self.mlp = SwiGLU(d, d_ff)

    def forward(self, x: Tensor, mask: Tensor | None, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.n1(x), mask, cos, sin)
        return x + self.mlp(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, d: int, n_layers: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.norm = RMSNorm(d)
        self.head_dim = d // n_heads

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        cos, sin = rope_cache(x.shape[1], self.head_dim, x.device)
        for blk in self.blocks:
            x = blk(x, mask, cos, sin)
        return self.norm(x)


def _mask_from_padding(key_pad: Tensor, causal: bool) -> Tensor:
    """Boolean SDPA mask [B,1,S,S] (True = may attend). Pad queries keep their own position."""
    b, s = key_pad.shape
    allow = ~key_pad[:, None, None, :]  # [B,1,1,S]
    allow = allow.expand(b, 1, s, s)
    if causal:
        allow = allow & torch.ones(s, s, dtype=torch.bool, device=key_pad.device).tril()[None, None]
    eye = torch.eye(s, dtype=torch.bool, device=key_pad.device)[None, None]
    return allow | eye


class IconSmith(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, de = cfg.d_model, cfg.enc_d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, d, padding_idx=PAD)
        # exemplar encoder (shares tok_emb, projected to its own width)
        self.enc_in = nn.Linear(d, de, bias=False)
        self.encoder = Transformer(de, cfg.enc_layers, cfg.enc_heads, int(de * 8 / 3) // 64 * 64,
                                   cfg.dropout)
        self.enc_out = nn.Linear(de, d, bias=False)
        # conditioning prefix
        self.stroke_emb = nn.Embedding(cfg.n_stroke_buckets, d)
        self.text_proj = nn.Linear(cfg.text_dim, d)
        self.null_style = nn.Parameter(torch.zeros(d))
        self.null_text = nn.Parameter(torch.zeros(d))
        self.prefix_type = nn.Embedding(3, d)  # 0 stroke, 1 text, 2 style
        # decoder
        self.decoder = Transformer(d, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.dropout)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        nn.init.normal_(self.null_style, std=0.02)
        nn.init.normal_(self.null_text, std=0.02)

    @property
    def prefix_len(self) -> int:
        return 2 + self.cfg.k

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode_style(self, exemplars: Tensor) -> Tensor:
        b, k, le = exemplars.shape
        flat = exemplars.reshape(b * k, le)
        pad = flat == PAD
        x = self.enc_in(self.tok_emb(flat))
        h = self.encoder(x, _mask_from_padding(pad, causal=False))
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.enc_out(pooled).view(b, k, -1)

    def _prefix(
        self,
        text_emb: Tensor,
        exemplars: Tensor,
        stroke_bucket: Tensor,
        drop_style: Tensor | None,
        drop_text: Tensor | None,
    ) -> Tensor:
        b = text_emb.shape[0]
        dev = text_emb.device
        if drop_style is None:
            p = self.cfg.p_drop_style if self.training else 0.0
            drop_style = torch.rand(b, device=dev) < p
        if drop_text is None:
            p = self.cfg.p_drop_text if self.training else 0.0
            drop_text = torch.rand(b, device=dev) < p
        style = self.encode_style(exemplars)  # [B,k,d]
        style = torch.where(drop_style[:, None, None], self.null_style.expand_as(style), style)
        text = self.text_proj(text_emb)  # [B,d]
        text = torch.where(drop_text[:, None], self.null_text.expand_as(text), text)
        stroke = self.stroke_emb(stroke_bucket)  # [B,d]
        types = self.prefix_type.weight
        prefix = torch.cat(
            [(stroke + types[0])[:, None], (text + types[1])[:, None], style + types[2]], dim=1
        )
        return prefix

    def forward(
        self,
        tokens: Tensor,
        text_emb: Tensor,
        exemplars: Tensor,
        stroke_bucket: Tensor,
        *,
        drop_style: Tensor | None = None,
        drop_text: Tensor | None = None,
    ) -> Tensor:
        prefix = self._prefix(text_emb, exemplars, stroke_bucket, drop_style, drop_text)
        inp = tokens[:, :-1]
        x = torch.cat([prefix, self.tok_emb(inp)], dim=1)
        # Padding is trailing-only, so causal attention already keeps pads from influencing
        # real positions; pad positions themselves are ignored by the loss.
        h = self.decoder(x, None)
        return self.head(h[:, self.prefix_len :])

    def loss(self, batch: dict) -> Tensor:
        b = batch
        logits = self(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"])
        target = batch["tokens"][:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1), ignore_index=PAD
        )
