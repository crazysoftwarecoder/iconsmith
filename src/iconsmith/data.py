"""Torch dataset over the processed .npz shards, with same-set exemplar sampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from iconsmith.dataset_build import set_file_name
from iconsmith.tokenizer import PAD

K = 8
EXEMPLAR_MAX_LEN = 256  # style is evident from the first subpaths; keeps the encoder cheap
SPLITS = ("train", "inset_test", "heldout")


def set_names(processed_root: Path) -> list[str]:
    return [s["name"] for s in json.loads((processed_root / "sets.json").read_text())]


class _SetShard:
    def __init__(self, root: Path, meta: dict) -> None:
        z = np.load(root / set_file_name(meta["name"]), allow_pickle=False)
        self.tokens = z["tokens"].astype(np.int64)
        self.offsets = z["offsets"]
        self.names = list(z["names"])
        self.text_emb = z["text_emb"].astype(np.float32)
        self.split = z["split"]
        self.meta = meta

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def item_tokens(self, i: int) -> np.ndarray:
        return self.tokens[self.offsets[i] : self.offsets[i + 1]]


class IconDataset(Dataset):
    def __init__(
        self,
        processed_root: Path,
        split: str,
        *,
        k: int = K,
        seed: int = 0,
        exemplar_max_len: int = EXEMPLAR_MAX_LEN,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.k = k
        self.exemplar_max_len = exemplar_max_len
        self.rng = np.random.default_rng(seed)
        metas = json.loads((processed_root / "sets.json").read_text())
        self.set_names = [m["name"] for m in metas]
        self.shards: dict[int, _SetShard] = {}
        self.items: list[tuple[int, int]] = []  # (set_idx, row)
        self.pool: dict[int, np.ndarray] = {}  # set_idx -> rows exemplars are drawn from
        for set_idx, meta in enumerate(metas):
            if meta["heldout"] != (split == "heldout"):
                continue
            shard = _SetShard(processed_root, meta)
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

    def __getitem__(self, i: int) -> dict:
        set_idx, row = self.items[i]
        shard = self.shards[set_idx]
        ex = self._sample_exemplars(set_idx, row)
        le = max(len(e) for e in ex)
        ex_t = torch.full((self.k, le), PAD, dtype=torch.long)
        for j, e in enumerate(ex):
            ex_t[j, : len(e)] = torch.from_numpy(e)
        return dict(
            tokens=torch.from_numpy(shard.item_tokens(row)),
            text_emb=torch.from_numpy(shard.text_emb[row]),
            exemplars=ex_t,
            stroke_bucket=int(shard.meta["stroke_bucket"]),
            set_idx=set_idx,
            name=shard.names[row],
        )


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
    return dict(
        tokens=tokens,
        exemplars=exemplars,
        text_emb=torch.stack([x["text_emb"] for x in batch]),
        stroke_bucket=torch.tensor([x["stroke_bucket"] for x in batch], dtype=torch.long),
        set_idx=torch.tensor([x["set_idx"] for x in batch], dtype=torch.long),
        names=[x["name"] for x in batch],
    )
