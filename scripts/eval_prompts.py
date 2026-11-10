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
from generate import (  # noqa: E402
    load_model,
    pick_exemplars,
    png,
    reference_svgs,
    retrieve_content,
    sample_best,
)

from iconsmith.data import ContentPool, K, _SetShard  # noqa: E402
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
    ap.add_argument("--best-of", type=int, default=1)
    ap.add_argument("--rerank-ref", action="store_true")
    ap.add_argument("--start-ref", action="store_true", help="force the reference's first point")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    metas = {m["name"]: m for m in json.loads((a.processed / "sets.json").read_text())}
    meta = metas[a.set]
    sw = meta["stroke_width_rel_median"]
    device = resolve_device("auto")
    model = load_model(a.ckpt, device)
    shard = _SetShard(a.processed, meta, model.canonical)
    refs = np.stack([rasterize(icon_to_svg(decode(shard.item_tokens(i).tolist()), sw), PX)
                     for i in range(len(shard))]).reshape(len(shard), -1)

    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=device).manual_seed(a.seed)
    exemplars, _ = pick_exemplars(shard, rng, K)
    emb = embed_names(list(WORDS))
    text = torch.from_numpy(emb).repeat_interleave(a.num, 0)
    need_pool = model.cfg.n_content or a.rerank_ref or a.start_ref
    pool = ContentPool(a.processed, model.canonical) if need_pool else None
    content, _ = retrieve_content(model, pool, emb, list(metas).index(a.set))
    if a.rerank_ref or a.start_ref:  # per word, so each word is reranked against its own reference
        ref_svgs = reference_svgs(pool, emb, list(metas).index(a.set), sw)
        toks = []
        for w_i in range(len(WORDS)):
            c = None if content is None else content[w_i : w_i + 1]
            toks += sample_best(model, text[w_i * a.num : w_i * a.num + 1], exemplars[None],
                                meta["stroke_bucket"], a.num, a.best_of, a.temperature, a.top_p,
                                device, gen, a.cfg, c, ref_svgs[w_i][1] if a.rerank_ref else None,
                                sw, ref_svgs[w_i][2] if a.start_ref else None)
    else:
        if content is not None:
            content = content.repeat_interleave(a.num, 0)
        toks = sample_best(model, text, exemplars[None], meta["stroke_bucket"], len(text),
                           a.best_of, a.temperature, a.top_p, device, gen, a.cfg, content)

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
    print(f"[eval] {a.ckpt} cfg {a.cfg} best-of {a.best_of} rerank-ref {a.rerank_ref} "
          f"start-ref {a.start_ref}: "
          f"{total}/{len(WORDS) * a.num} hits "
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
