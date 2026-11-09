"""Prompt-recognisability eval: does "heart" come out as a heart?

For each prompt word, sample n icons in a set's style, rasterise them, and find the nearest icon
of that set (by raster IoU). A sample is a hit when the nearest icon's name starts with the word.
Prints per-word hits and the overall hit rate; writes a contact sheet.

Usage: uv run python scripts/eval_prompts.py --ckpt runs/v4/ckpt_best.pt [--cfg 2] [-n 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from generate import load_model, pick_exemplars, png, sample  # noqa: E402

from iconsmith.data import K, _SetShard  # noqa: E402
from iconsmith.render import icon_to_svg, rasterize  # noqa: E402
from iconsmith.text_embed import embed_names  # noqa: E402
from iconsmith.tokenizer import decode  # noqa: E402
from iconsmith.train import resolve_device  # noqa: E402

WORDS = ("heart house star coffee key cloud camera bell lock sun moon search user mail phone "
         "trash calendar folder umbrella anchor").split()
PX = 48


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--set", default="lucide")
    ap.add_argument("-n", "--num", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    meta = {m["name"]: m for m in json.loads((a.processed / "sets.json").read_text())}[a.set]
    sw = meta["stroke_width_rel_median"]
    device = resolve_device("auto")
    model = load_model(a.ckpt, device)
    shard = _SetShard(a.processed, meta, model.canonical)
    refs = np.stack([rasterize(icon_to_svg(decode(shard.item_tokens(i).tolist()), sw), PX)
                     for i in range(len(shard))]).reshape(len(shard), -1)

    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=device).manual_seed(a.seed)
    exemplars, _ = pick_exemplars(shard, rng, K)
    text = torch.from_numpy(embed_names(list(WORDS))).repeat_interleave(a.num, 0)
    toks = sample(model, text, exemplars[None], meta["stroke_bucket"], len(text),
                  a.temperature, a.top_p, device, gen, a.cfg)

    hits, rows = {}, []
    for w_i, word in enumerate(WORDS):
        row, h = [], 0
        for s in range(a.num):
            t = toks[w_i * a.num + s]
            if t is None:
                continue
            svg = icon_to_svg(decode(t), sw)
            r = rasterize(svg, PX).reshape(1, -1)
            inter = (refs & r).sum(1)
            union = (refs | r).sum(1)
            nn = int(np.argmax(inter / np.maximum(union, 1)))
            h += shard.names[nn].startswith(word)
            row.append(png(svg, 64))
        hits[word] = h
        rows.append(row)
    total = sum(hits.values())
    print(json.dumps(hits))
    print(f"[eval] {a.ckpt} cfg {a.cfg}: {total}/{len(WORDS) * a.num} hits "
          f"({total / (len(WORDS) * a.num):.0%})")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        sheet = Image.new("RGBA", (a.num * 72, len(rows) * 72), "white")
        for r, row in enumerate(rows):
            for c, im in enumerate(row):
                sheet.alpha_composite(im, (c * 72 + 4, r * 72 + 4))
        sheet.save(a.out)


if __name__ == "__main__":
    main()
