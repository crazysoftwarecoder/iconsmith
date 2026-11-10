"""Torch dataset over the processed .npz shards, with same-set exemplar sampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from iconsmith.dataset_build import set_file_name
from iconsmith.render import alpha, icon_to_svg
from iconsmith.tokenizer import COORD_OFFSET, GRID, PAD, C, L, M, canonical_order, decode, encode

K = 8
EXEMPLAR_MAX_LEN = 256  # style is evident from the first subpaths; keeps the encoder cheap
RASTER_STROKE = 0.083  # stroke width content pictures are drawn with, whatever their set
CONTENT_MAX_LEN = 384  # content exemplars carry the shape, so keep more of them
SAME_RENDER_BONUS = 0.15
SPLITS = ("train", "inset_test", "heldout")


def set_names(processed_root: Path) -> list[str]:
    return [s["name"] for s in json.loads((processed_root / "sets.json").read_text())]


def _xy_masks(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masks of x and y coord tokens (coords alternate x, y after each M/L/C)."""
    is_coord = t >= COORD_OFFSET
    cmd = (t == M) | (t == L) | (t == C)
    n_coord = np.cumsum(is_coord)
    # coords seen so far at the most recent command token
    base = np.maximum.accumulate(np.where(cmd, n_coord, 0))
    is_x = is_coord & ((n_coord - base) % 2 == 1)
    return is_x, is_coord & ~is_x


def affine(tokens: np.ndarray, scale: float = 1.0, dx: float = 0.0, dy: float = 0.0,
           flip: bool = False) -> np.ndarray:
    """Scale about the grid centre, optionally mirror horizontally, then shift; clipped to grid."""
    t = tokens.copy()
    is_x, is_y = _xy_masks(t)
    c = (GRID - 1) / 2
    for mask, sign, d in ((is_x, -1.0 if flip else 1.0, dx), (is_y, 1.0, dy)):
        v = (t[mask] - COORD_OFFSET - c) * scale * sign + c + d
        t[mask] = COORD_OFFSET + np.clip(np.rint(v), 0, GRID - 1).astype(t.dtype)
    return t


def warp(tokens: np.ndarray, rng: np.random.Generator, sigma: float, j: float = 0.1) -> np.ndarray:
    """A redrawn-by-hand copy: small random scale/shift plus a smooth per-axis wobble of about
    ``sigma`` grid units. The wobble is a function of the coordinate value, so coincident points
    (closed subpaths, shared corners) stay coincident."""
    t = affine(tokens, float(rng.uniform(1 - j, 1 + j / 2)),
               float(rng.uniform(-j, j)) * GRID / 2, float(rng.uniform(-j, j)) * GRID / 2)
    grid = np.arange(GRID)
    for mask in _xy_masks(t):
        f, ph = rng.uniform(0.5, 2.5, 2), rng.uniform(0, 2 * np.pi, 2)
        wob = sigma * (np.sin(2 * np.pi * f[0] * grid / GRID + ph[0])
                       + np.sin(2 * np.pi * f[1] * grid / GRID + ph[1])) / np.sqrt(2)
        lut = np.clip(np.rint(grid + wob), 0, GRID - 1).astype(t.dtype)
        t[mask] = COORD_OFFSET + lut[t[mask] - COORD_OFFSET]
    return t


def flip_x(tokens: np.ndarray) -> np.ndarray:
    """Mirror an icon horizontally: each x coord token -> GRID-1-x."""
    return affine(tokens, flip=True)


def recanonical(tokens: np.ndarray, level: int) -> np.ndarray:
    """Re-apply canonical order (level 1) / starts (level 2) to one encoded icon."""
    icon = canonical_order(decode(tokens.tolist()), starts=level >= 2)
    return np.asarray(encode(icon), dtype=np.int64)


class _SetShard:
    def __init__(self, root: Path, meta: dict, canonical: int = 0) -> None:
        """canonical: 0 = as stored, 1 = sorted subpaths, 2 = also canonical start points."""
        z = np.load(root / set_file_name(meta["name"]), allow_pickle=False)
        self.tokens = z["tokens"].astype(np.int64)
        self.offsets = z["offsets"]
        if canonical:  # same tokens, subpaths re-sorted (lengths unchanged, so offsets still hold)
            self.tokens = np.concatenate([
                np.asarray(encode(canonical_order(decode(
                    self.tokens[self.offsets[i] : self.offsets[i + 1]].tolist()),
                    starts=canonical >= 2)), dtype=np.int64)
                for i in range(len(self.offsets) - 1)
            ] or [self.tokens])
        self.names = list(z["names"])
        self.text_emb = z["text_emb"].astype(np.float32)
        self.split = z["split"]
        self.meta = meta

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def item_tokens(self, i: int) -> np.ndarray:
        return self.tokens[self.offsets[i] : self.offsets[i + 1]]


class ContentPool:
    """Training icons of all training sets, searchable by name embedding.

    Content exemplars for an icon are the icons whose names are closest to it (MiniLM cosine),
    taken from *other libraries* (sibling sets like tabler/outline vs tabler/filled share their
    geometry, which would teach copying), so the model learns to redraw a shape in a new style.
    """

    def __init__(self, processed_root: Path, canonical: int = 0,
                 shards: dict[int, _SetShard] | None = None) -> None:
        metas = json.loads((processed_root / "sets.json").read_text())
        shards = dict(shards or {})
        keys, embs = [], []
        for set_idx, meta in enumerate(metas):
            if meta["heldout"]:
                continue
            if set_idx not in shards:
                shards[set_idx] = _SetShard(processed_root, meta, canonical)
            sh = shards[set_idx]
            rows = np.flatnonzero(sh.split == 0)
            keys.extend((set_idx, int(r)) for r in rows)
            embs.append(sh.text_emb[rows])
        self.shards = shards
        self.set_names = [m["name"] for m in metas]
        libs = sorted({m["library"] for m in metas})
        self.set_lib = np.array([libs.index(m["library"]) for m in metas])
        self.keys = np.array(keys, dtype=np.int64).reshape(-1, 2)
        self.key_lib = self.set_lib[self.keys[:, 0]] if len(self.keys) else np.zeros(0, int)
        self.set_stroke = np.array([m["render"] == "stroke" for m in metas])
        self.key_stroke = self.set_stroke[self.keys[:, 0]] if len(self.keys) else np.zeros(0, bool)
        self.emb = np.concatenate(embs) if embs else np.zeros((0, 384), np.float32)

    def nearest(self, emb: np.ndarray, target_set: np.ndarray, top: int,
                chunk: int = 2048) -> np.ndarray:
        """Indices [Q, top] into ``keys`` of the nearest names outside each target's library.

        Icons drawn the same way as the target (stroke vs fill) get +SAME_RENDER_BONUS, so an
        exact-name match in the other render mode still wins over a merely related name.
        """
        exclude = self.set_lib[target_set]
        stroke = self.set_stroke[target_set]
        out = np.zeros((len(emb), top), dtype=np.int64)
        for i in range(0, len(emb), chunk):
            sims = emb[i : i + chunk] @ self.emb.T
            sims += SAME_RENDER_BONUS * (self.key_stroke[None, :] == stroke[i : i + chunk, None])
            sims[self.key_lib[None, :] == exclude[i : i + chunk, None]] = -np.inf
            part = np.argpartition(-sims, top, axis=1)[:, :top]
            order = np.argsort(-np.take_along_axis(sims, part, 1), axis=1)
            out[i : i + chunk] = np.take_along_axis(part, order, 1)
        return out

    def tokens(self, idx: int, max_len: int | None = CONTENT_MAX_LEN) -> np.ndarray:
        set_idx, row = self.keys[idx]
        return self.shards[int(set_idx)].item_tokens(int(row))[:max_len]

    def describe(self, idx: int) -> str:
        set_idx, row = self.keys[idx]
        return f"{self.set_names[int(set_idx)]}:{self.shards[int(set_idx)].names[int(row)]}"


class IconDataset(Dataset):
    def __init__(
        self,
        processed_root: Path,
        split: str,
        *,
        k: int = K,
        seed: int = 0,
        exemplar_max_len: int = EXEMPLAR_MAX_LEN,
        p_flip: float = 0.0,
        jitter: float = 0.0,
        canonical: int = 0,
        n_content: int = 0,
        content_top: int = 4,
        p_self_content: float = 0.0,
        self_noise: float = 4.0,
        content_raster: int = 0,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.k = k
        self.exemplar_max_len = exemplar_max_len
        self.p_flip = p_flip  # horizontal-flip augmentation (target and exemplars together)
        self.canonical = canonical
        self.jitter = jitter  # random scale/shift augmentation strength (0 = off)
        self.rng = np.random.default_rng(seed)
        metas = json.loads((processed_root / "sets.json").read_text())
        self.set_names = [m["name"] for m in metas]
        self.shards: dict[int, _SetShard] = {}
        self.items: list[tuple[int, int]] = []  # (set_idx, row)
        self.pool: dict[int, np.ndarray] = {}  # set_idx -> rows exemplars are drawn from
        for set_idx, meta in enumerate(metas):
            if meta["heldout"] != (split == "heldout"):
                continue
            shard = _SetShard(processed_root, meta, canonical)
            if len(shard) == 0:
                continue
            if split == "train":
                rows = np.flatnonzero(shard.split == 0)
            elif split == "inset_test":
                rows = np.flatnonzero(shard.split == 1)
            else:
                rows = np.arange(len(shard))
            if len(rows) == 0:
                continue
            self.shards[set_idx] = shard
            # Exemplars never come from the in-set test rows of a training set.
            self.pool[set_idx] = np.flatnonzero(shard.split == 0) if not meta["heldout"] else rows
            self.items.extend((set_idx, int(r)) for r in rows)
        self.n_content = n_content
        # Cross-library neighbours rarely line up with the target, so on their own the model
        # learns to ignore content. A warped copy of the target teaches it to read the shape.
        self.p_self_content = p_self_content
        self.self_noise = self_noise
        self.content_raster = content_raster  # >0: content is a picture this many px square
        if n_content:
            self.content = ContentPool(processed_root, canonical, self.shards)
            emb = np.stack([self.shards[s].text_emb[r] for s, r in self.items])
            sets = np.array([s for s, _ in self.items])
            # sampled from the top few each time, as augmentation; inference takes the nearest
            self.content_nbrs = self.content.nearest(emb, sets, max(content_top, n_content))

    def __len__(self) -> int:
        return len(self.items)

    def keys(self) -> list[tuple[str, str]]:
        return [(self.set_names[s], self.shards[s].names[r]) for s, r in self.items]

    def set_size(self, set_idx: int) -> int:
        return len(self.pool[set_idx])

    def _sample_exemplars(self, set_idx: int, row: int) -> list[np.ndarray]:
        pool = self.pool[set_idx]
        candidates = pool[pool != row]
        if len(candidates) == 0:
            candidates = pool
        replace = len(candidates) < self.k
        picks = self.rng.choice(candidates, size=self.k, replace=replace)
        shard = self.shards[set_idx]
        return [shard.item_tokens(int(p))[: self.exemplar_max_len] for p in picks]

    def _augment(self, ex: list[np.ndarray], tokens: np.ndarray, flip: bool,
                 content: list[np.ndarray]):
        """One random transform shared by the target and its exemplars, so style stays consistent.
        The shift is limited so the target icon stays on the grid."""
        scale, dx, dy = 1.0, 0.0, 0.0
        if self.jitter:
            j = self.jitter
            scale = float(self.rng.uniform(1 - j, 1 + j / 3))
            is_x, is_y = _xy_masks(tokens)
            c = (GRID - 1) / 2
            shifts = []
            for mask in (is_x, is_y):
                v = (tokens[mask] - COORD_OFFSET - c) * scale + c
                lo, hi = -float(v.min()), (GRID - 1) - float(v.max())  # room left on the grid
                # shift up to j*GRID, but never push the target off the grid
                d = self.rng.uniform(-j, j) * GRID
                shifts.append(float(np.clip(d, lo, hi)) if lo <= hi else 0.0)
            dx, dy = shifts
        return ([affine(e, scale, dx, dy, flip) for e in ex],
                affine(tokens, scale, dx, dy, flip),
                [affine(c, scale, dx, dy, flip) for c in content])

    def __getitem__(self, i: int) -> dict:
        set_idx, row = self.items[i]
        shard = self.shards[set_idx]
        ex = self._sample_exemplars(set_idx, row)
        tokens = shard.item_tokens(row)
        content = []
        if self.n_content:
            picks = self.rng.choice(self.content_nbrs[i], size=self.n_content, replace=False)
            content = [self.content.tokens(int(p), max_len=None) for p in picks]
        flip = bool(self.p_flip) and self.rng.random() < self.p_flip
        if flip or self.jitter:
            ex, tokens, content = self._augment(ex, tokens, flip, content)
            if self.canonical:  # a mirror moves the top-left start/order; restore the convention
                tokens = recanonical(tokens, self.canonical)
                content = [recanonical(c, self.canonical) for c in content]
        if self.n_content and self.p_self_content and self.rng.random() < self.p_self_content:
            c = warp(tokens, self.rng, self.self_noise)
            content[0] = recanonical(c, self.canonical) if self.canonical else c
        if self.content_raster:
            content = [content_picture(c, self.content_raster) for c in content]
        else:
            content = [c[:CONTENT_MAX_LEN] for c in content]
        le = max(len(e) for e in ex)
        ex_t = torch.full((self.k, le), PAD, dtype=torch.long)
        for j, e in enumerate(ex):
            ex_t[j, : len(e)] = torch.from_numpy(e)
        out = dict(
            tokens=torch.from_numpy(tokens),
            text_emb=torch.from_numpy(shard.text_emb[row]),
            exemplars=ex_t,
            stroke_bucket=int(shard.meta["stroke_bucket"]),
            set_idx=set_idx,
            name=shard.names[row],
        )
        if self.n_content:
            out["content"] = (torch.from_numpy(np.stack(content)) if self.content_raster
                              else pad_stack(content))
        return out


def content_picture(tokens: np.ndarray, px: int) -> np.ndarray:
    """A content icon drawn as a px x px coverage map (the global plan for raster content)."""
    return alpha(icon_to_svg(decode(tokens.tolist()), RASTER_STROKE), px)


def pad_stack(seqs: list[np.ndarray]) -> torch.Tensor:
    """[n, L] PAD-padded long tensor from variable-length token arrays."""
    t = torch.full((len(seqs), max(len(x) for x in seqs)), PAD, dtype=torch.long)
    for j, x in enumerate(seqs):
        t[j, : len(x)] = torch.as_tensor(x)
    return t


def collate(batch: list[dict]) -> dict:
    b = len(batch)
    lmax = max(len(x["tokens"]) for x in batch)
    k = batch[0]["exemplars"].shape[0]
    le = max(x["exemplars"].shape[1] for x in batch)
    tokens = torch.full((b, lmax), PAD, dtype=torch.long)
    exemplars = torch.full((b, k, le), PAD, dtype=torch.long)
    for i, x in enumerate(batch):
        tokens[i, : len(x["tokens"])] = x["tokens"]
        exemplars[i, :, : x["exemplars"].shape[1]] = x["exemplars"]
    out = dict(
        tokens=tokens,
        exemplars=exemplars,
        text_emb=torch.stack([x["text_emb"] for x in batch]),
        stroke_bucket=torch.tensor([x["stroke_bucket"] for x in batch], dtype=torch.long),
        set_idx=torch.tensor([x["set_idx"] for x in batch], dtype=torch.long),
        names=[x["name"] for x in batch],
    )
    if "content" in batch[0] and batch[0]["content"].is_floating_point():  # pictures
        out["content"] = torch.stack([x["content"] for x in batch])
    elif "content" in batch[0]:
        c, lc =batch[0]["content"].shape[0], max(x["content"].shape[1] for x in batch)
        out["content"] = torch.full((b, c, lc), PAD, dtype=torch.long)
        for i, x in enumerate(batch):
            out["content"][i, :, : x["content"].shape[1]] = x["content"]
    return out
