"""Set-conditioned icon transformer.

Decoder-only transformer over path tokens, conditioned on a prefix of
``[stroke-width bucket] [name embedding] [K exemplar style vectors]``.
Exemplars are encoded by a small bidirectional transformer that shares the token embedding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from iconsmith.data import K
from iconsmith.dataset_build import N_STROKE_BUCKETS
from iconsmith.tokenizer import COORD_OFFSET, GRID, PAD, VOCAB_SIZE


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
    # v1 checkpoints: 1 token, unscaled. v2: name spread over several prefix slots, rescaled.
    n_text_tokens: int = 1
    text_scale: float = 1.0
    # retrieval: icons with the nearest names from other sets, each pooled into content_slots
    n_content: int = 0
    content_slots: int = 16
    p_drop_content: float = 0.2
    # "slots": each content icon pooled to content_slots vectors (v7; tested: ignored by the
    # decoder). "tokens": every encoded content token goes into the prefix, so it can be copied.
    content_mode: str = "slots"
    # "raster": each content icon is a raster_px x raster_px picture cut into patch tokens. A
    # picture lines up with the coordinates whatever the source library's path order, so it can
    # act as a global plan (token content from another library was ignored, v8-v10).
    raster_px: int = 32
    raster_patch: int = 4
    # >0: coordinate tokens get sin/cos features of their value (this many frequencies), and
    # raster patches the same features of their centre, so "x = 74" can find "column ~74"
    coord_fourier: int = 0


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

    def forward(self, x: Tensor, mask: Tensor | None, cos: Tensor, sin: Tensor,
                cache: dict | None = None) -> Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        causal = mask is None
        if cache is not None:  # incremental decoding: append to and attend over past keys
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
                assert s == 1, "cached decoding feeds one token at a time"
                causal = False  # a single new query may see every cached key
            cache["k"], cache["v"] = k, v
        # mask=None means causal via the fused kernel (no materialised S x S mask).
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal,
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

    def forward(self, x: Tensor, mask: Tensor | None, cos: Tensor, sin: Tensor,
                cache: dict | None = None) -> Tensor:
        x = x + self.attn(self.n1(x), mask, cos, sin, cache)
        return x + self.mlp(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, d: int, n_layers: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.norm = RMSNorm(d)
        self.head_dim = d // n_heads

    def forward(self, x: Tensor, mask: Tensor | None, cache: list[dict] | None = None,
                pos: int = 0) -> Tensor:
        cos, sin = rope_cache(pos + x.shape[1], self.head_dim, x.device)
        cos, sin = cos[pos:], sin[pos:]
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask, cos, sin, None if cache is None else cache[i])
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
        if cfg.coord_fourier:
            # 1 to GRID/4 cycles across the grid
            n_f = cfg.coord_fourier
            freqs = 2 * math.pi * torch.logspace(0, math.log2(GRID / 4), n_f, 2) / GRID
            self.register_buffer("fourier_freqs", freqs, persistent=False)
            self.coord_ff = nn.Linear(2 * cfg.coord_fourier, d, bias=False)
            if cfg.n_content and cfg.content_mode == "raster":
                self.patch_ff_x = nn.Linear(2 * cfg.coord_fourier, d, bias=False)
                self.patch_ff_y = nn.Linear(2 * cfg.coord_fourier, d, bias=False)
        # exemplar encoder (shares tok_emb, projected to its own width)
        self.enc_in = nn.Linear(d, de, bias=False)
        self.encoder = Transformer(de, cfg.enc_layers, cfg.enc_heads, int(de * 8 / 3) // 64 * 64,
                                   cfg.dropout)
        self.enc_out = nn.Linear(de, d, bias=False)
        # conditioning prefix
        self.stroke_emb = nn.Embedding(cfg.n_stroke_buckets, d)
        # unit-norm MiniLM vectors are ~20x smaller per-dim than token embeddings (text_scale),
        # and one slot is easily drowned out by style (n_text_tokens)
        self.text_proj = nn.Linear(cfg.text_dim, cfg.n_text_tokens * d)
        self.null_style = nn.Parameter(torch.zeros(d))
        self.null_text = nn.Parameter(torch.zeros(cfg.n_text_tokens, d))
        self.prefix_type = nn.Embedding(4 if cfg.n_content else 3, d)  # stroke/text/style/content
        if cfg.n_content and cfg.content_mode == "slots":
            # learned queries keep a coarse layout of each content icon (a mean would lose it)
            self.content_queries = nn.Parameter(torch.randn(cfg.content_slots, de) * 0.02)
            self.content_attn = nn.MultiheadAttention(de, cfg.enc_heads, batch_first=True)
        if cfg.n_content and cfg.content_mode == "raster":
            n_patch = (cfg.raster_px // cfg.raster_patch) ** 2
            self.raster_in = nn.Sequential(nn.Linear(cfg.raster_patch ** 2, d), nn.GELU(),
                                           nn.Linear(d, d))
            self.raster_pos = nn.Parameter(torch.randn(n_patch, d) * 0.02)
        if cfg.n_content:
            self.content_out = nn.Linear(de, d, bias=False)
            self.null_content = nn.Parameter(torch.randn(d) * 0.02)
        # decoder
        self.decoder = Transformer(d, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.dropout)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        nn.init.normal_(self.null_style, std=0.02)
        nn.init.normal_(self.null_text, std=0.02)

    @property
    def prefix_len(self) -> int:
        """Prefix length without content in "tokens" mode (there it depends on the content)."""
        c = self.cfg
        slots = {"slots": c.content_slots, "tokens": 0,
                 "raster": (c.raster_px // c.raster_patch) ** 2}[c.content_mode]
        return 1 + c.n_text_tokens + c.k + c.n_content * slots

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode_style(self, exemplars: Tensor) -> Tensor:
        b, k, le = exemplars.shape
        flat = exemplars.reshape(b * k, le)
        pad = flat == PAD
        x = self.enc_in(self.embed(flat))
        h = self.encoder(x, _mask_from_padding(pad, causal=False))
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.enc_out(pooled).view(b, k, -1)

    def _fourier(self, v: Tensor) -> Tensor:
        a = v.float()[..., None] * self.fourier_freqs
        return torch.cat([a.sin(), a.cos()], -1)

    def embed(self, tokens: Tensor) -> Tensor:
        """Token embedding, plus value features for coordinates when coord_fourier is set."""
        e = self.tok_emb(tokens)
        if not self.cfg.coord_fourier:
            return e
        is_coord = (tokens >= COORD_OFFSET)[..., None]
        feat = self.coord_ff(self._fourier((tokens - COORD_OFFSET).clamp(min=0)).to(e.dtype))
        return e + feat * is_coord

    def encode_raster(self, raster: Tensor) -> Tensor:
        """[B, C, px, px] alpha in 0..1 -> [B, C * patches, d] patch tokens."""
        b, c, px, _ = raster.shape
        q = self.cfg.raster_patch
        g = px // q
        x = raster.reshape(b * c, g, q, g, q).permute(0, 1, 3, 2, 4).reshape(b * c, g * g, q * q)
        h = self.raster_in(x.to(self.raster_pos.dtype)) + self.raster_pos
        if self.cfg.coord_fourier:  # patch centres in grid units, same features as coordinates
            centre = (torch.arange(g, device=raster.device) + 0.5) * GRID / g - 0.5
            fy, fx = self._fourier(centre)[:, None], self._fourier(centre)[None, :]
            pos = self.patch_ff_y(fy.expand(g, g, -1)) + self.patch_ff_x(fx.expand(g, g, -1))
            h = h + pos.reshape(g * g, -1)
        return h.reshape(b, c * g * g, -1)

    def encode_content(self, content: Tensor) -> Tensor:
        if self.cfg.content_mode == "raster":
            return self.encode_raster(content)
        b, c, le = content.shape
        flat = content.reshape(b * c, le)
        pad = flat == PAD
        pad[:, 0] = False  # an all-pad row would make attention NaN; it is replaced by null anyway
        h = self.encoder(self.enc_in(self.embed(flat)), _mask_from_padding(pad, causal=False))
        if self.cfg.content_mode == "tokens":
            # pads become one constant learned vector, which the decoder learns to ignore
            out = torch.where(pad[..., None], self.null_content.to(h.dtype), self.content_out(h))
            return out.reshape(b, c * le, -1)
        q = self.content_queries.expand(b * c, -1, -1).to(h.dtype)
        out, _ = self.content_attn(q, h, h, key_padding_mask=pad, need_weights=False)
        return self.content_out(out).reshape(b, c * self.cfg.content_slots, -1)

    def _prefix(
        self,
        text_emb: Tensor,
        exemplars: Tensor,
        stroke_bucket: Tensor,
        drop_style: Tensor | None,
        drop_text: Tensor | None,
        content: Tensor | None = None,
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
        text = self.text_proj(text_emb * self.cfg.text_scale).view(b, self.cfg.n_text_tokens, -1)
        text = torch.where(drop_text[:, None, None], self.null_text.expand_as(text), text)
        stroke = self.stroke_emb(stroke_bucket)  # [B,d]
        types = self.prefix_type.weight
        parts = [(stroke + types[0])[:, None], text + types[1], style + types[2]]
        if self.cfg.n_content:
            # content is part of "what to draw": dropped with the name, plus on its own
            drop_c = drop_text
            if self.training:
                drop_c = drop_c | (torch.rand(b, device=dev) < self.cfg.p_drop_content)
            if content is None:
                n = {"slots": self.cfg.content_slots, "tokens": 1,
                     "raster": (self.cfg.raster_px // self.cfg.raster_patch) ** 2}[
                         self.cfg.content_mode]
                cont = self.null_content.expand(b, self.cfg.n_content * n, -1)
            else:
                enc = self.encode_content(content)
                cont = torch.where(drop_c[:, None, None], self.null_content.to(enc.dtype), enc)
            parts.append(cont + types[3])
        return torch.cat(parts, dim=1)

    def forward(
        self,
        tokens: Tensor,
        text_emb: Tensor,
        exemplars: Tensor,
        stroke_bucket: Tensor,
        *,
        drop_style: Tensor | None = None,
        drop_text: Tensor | None = None,
        content: Tensor | None = None,
    ) -> Tensor:
        prefix = self._prefix(text_emb, exemplars, stroke_bucket, drop_style, drop_text, content)
        inp = tokens[:, :-1]
        x = torch.cat([prefix, self.embed(inp)], dim=1)
        # Padding is trailing-only, so causal attention already keeps pads from influencing
        # real positions; pad positions themselves are ignored by the loss.
        h = self.decoder(x, None)
        return self.head(h[:, prefix.shape[1] :])

    @torch.no_grad()
    def start_decode(self, tokens: Tensor, text_emb: Tensor, exemplars: Tensor,
                     stroke_bucket: Tensor, drop_text: Tensor,
                     content: Tensor | None = None) -> tuple[Tensor, list[dict]]:
        """Run prefix + ``tokens`` once; returns next-token logits and a KV cache."""
        no_drop = torch.zeros_like(drop_text)
        prefix = self._prefix(text_emb, exemplars, stroke_bucket, no_drop, drop_text, content)
        cache: list[dict] = [{} for _ in self.decoder.blocks]
        h = self.decoder(torch.cat([prefix, self.embed(tokens)], 1), None, cache)
        return self.head(h[:, -1]), cache

    @torch.no_grad()
    def step_decode(self, tok: Tensor, cache: list[dict]) -> Tensor:
        """Feed one token per row (shape [B]); returns next-token logits."""
        pos = cache[0]["k"].shape[2]
        h = self.decoder(self.embed(tok[:, None]), None, cache, pos)
        return self.head(h[:, -1])

    def loss(self, batch: dict) -> Tensor:
        b = batch
        logits = self(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"],
                      content=b.get("content"))
        target = batch["tokens"][:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1), ignore_index=PAD
        )
