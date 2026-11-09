"""Generate icons from a trained checkpoint: style from an existing set, content from a name.

Usage:
  uv run python scripts/generate.py --list-sets
  uv run python scripts/generate.py --set lucide --name "coffee cup" --name rocket -n 4
  uv run python scripts/generate.py --set iconoir/regular --name "shopping cart" --temperature 0.8

Writes one SVG per sample plus the K style exemplars and a PNG contact sheet to --out.
Decoding is grammar-constrained, so every finished sample is a valid SVG path.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import resvg_py
import torch
from PIL import Image

from iconsmith.data import EXEMPLAR_MAX_LEN, K, _SetShard
from iconsmith.model import IconSmith, ModelConfig
from iconsmith.render import icon_to_svg
from iconsmith.text_embed import embed_names
from iconsmith.tokenizer import (
    BOS,
    COORD_OFFSET,
    MAX_TOKENS,
    PAD,
    VOCAB_SIZE,
    GrammarState,
    decode,
)
from iconsmith.train import resolve_device


def load_model(ckpt_path: Path, device: str) -> IconSmith:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = IconSmith(ModelConfig(**ckpt["model_cfg"]))
    sd = ckpt["state_dict"]
    if sd["null_text"].ndim == 1:  # v1 checkpoints stored a single text slot without the slot dim
        sd["null_text"] = sd["null_text"][None]
    model.load_state_dict(sd)
    # exemplars must be ordered the way the model was trained on them
    model.canonical = ckpt.get("train_cfg", {}).get("canonical") == "True"
    print(f"[gen] loaded {ckpt_path} (step {ckpt.get('step')})")
    return model.to(device).eval()


def pick_exemplars(shard: _SetShard, rng: np.random.Generator, k: int) -> tuple[torch.Tensor, list]:
    pool = np.arange(len(shard)) if shard.meta["heldout"] else np.flatnonzero(shard.split == 0)
    picks = rng.choice(pool, size=k, replace=len(pool) < k)
    ex = [shard.item_tokens(int(p))[:EXEMPLAR_MAX_LEN] for p in picks]
    t = torch.full((k, max(len(e) for e in ex)), PAD, dtype=torch.long)
    for j, e in enumerate(ex):
        t[j, : len(e)] = torch.from_numpy(e)
    return t, [(shard.names[int(p)], shard.item_tokens(int(p)).tolist()) for p in picks]


@torch.no_grad()
def sample(model, text_emb, exemplars, stroke_bucket, n, temperature, top_p, device, gen, cfg=1.0,
           guide_structure=True):
    """Autoregressive grammar-constrained sampling of n icons (KV-cached).

    text_emb is [1, D] (shared) or [n, D]. With cfg != 1 the name-less prediction is decoded in
    the same batch and the logits are pushed away from it (classifier-free guidance).
    """
    text_emb = text_emb.expand(n, -1).to(device)
    exemplars = exemplars.expand(n, -1, -1).to(device)
    bucket = torch.full((n,), stroke_bucket, dtype=torch.long, device=device)
    drop = torch.zeros(n, dtype=torch.bool, device=device)
    rows = 2 if cfg != 1.0 else 1
    if rows == 2:
        text_emb, exemplars, bucket = (torch.cat([x, x]) for x in (text_emb, exemplars, bucket))
        drop = torch.cat([drop, ~drop])
    grammars = [GrammarState() for _ in range(n)]
    for g in grammars:
        g.step(BOS)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    seqs = [[BOS] for _ in range(n)]
    bos = torch.full((rows * n, 1), BOS, dtype=torch.long, device=device)
    pad_only = np.eye(VOCAB_SIZE, dtype=bool)[PAD]
    with torch.autocast(device_type=device.split(":")[0], dtype=dtype):
        out, cache = model.start_decode(bos, text_emb, exemplars, bucket, drop)
        for _ in range(MAX_TOKENS - 1):
            if all(g.done for g in grammars):
                break
            logits = out.float()
            if rows == 2:
                cond, null = torch.log_softmax(logits[:n], -1), torch.log_softmax(logits[n:], -1)
                logits = null + cfg * (cond - null)
                if not guide_structure:  # leave EOS/Z/M unguided (tested: worse, 11% vs 16% on v2)
                    logits[:, :COORD_OFFSET] = cond[:, :COORD_OFFSET]
            allowed = torch.from_numpy(np.stack([
                pad_only if g.done else g.allowed() for g in grammars])).to(device)
            logits = logits.masked_fill(~allowed, float("-inf")) / max(temperature, 1e-5)
            probs = torch.softmax(logits, -1)
            if top_p < 1.0:
                sp, si = probs.sort(-1, descending=True)
                keep = sp.cumsum(-1) - sp < top_p
                probs = torch.zeros_like(probs).scatter(-1, si, sp * keep)
                probs = probs / probs.sum(-1, keepdim=True)
            nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            for i, g in enumerate(grammars):
                if not g.done:
                    g.step(int(nxt[i]))
                    seqs[i].append(int(nxt[i]))
            out = model.step_decode(nxt.repeat(rows), cache)
    return [s if g.done else None for s, g in zip(seqs, grammars, strict=True)]


def png(svg: str, px: int) -> Image.Image:
    data = resvg_py.svg_to_bytes(svg_string=svg, width=px, height=px)
    return Image.open(io.BytesIO(bytes(data))).convert("RGBA")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=Path("runs/v1/ckpt_best.pt"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--set", help="icon set to copy the style from (see --list-sets)")
    ap.add_argument("--name", action="append", default=[], help="icon name; repeat for several")
    ap.add_argument("-n", "--num", type=int, default=4, help="samples per name")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--cfg", type=float, default=1.0, help="name guidance; 1 = off, 2-5 = stronger")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/samples"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--list-sets", action="store_true")
    a = ap.parse_args()

    metas = {m["name"]: m for m in json.loads((a.processed / "sets.json").read_text())}
    if a.list_sets or not a.set or not a.name:
        print("available sets (name, icons, held out of training):")
        for m in metas.values():
            print(f"  {m['name']:<24} {m['n_icons']:>5}  {'held-out' if m['heldout'] else ''}")
        if not a.list_sets:
            ap.error("--set and at least one --name are required")
        return
    if a.set not in metas:
        ap.error(f"unknown set {a.set!r}; run with --list-sets")

    device = resolve_device(a.device)
    model = load_model(a.ckpt, device)
    meta = metas[a.set]
    shard = _SetShard(a.processed, meta, model.canonical)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=device).manual_seed(a.seed)
    exemplars, ex_info = pick_exemplars(shard, rng, K)
    sw = meta["stroke_width_rel_median"]

    out = a.out / a.set.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    rows = [[png(icon_to_svg(decode(t), sw), 64) for _, t in ex_info]]
    for j, (ex_name, toks) in enumerate(ex_info):
        (out / f"exemplar_{j}_{ex_name}.svg").write_text(icon_to_svg(decode(toks), sw))
    print(f"[gen] style exemplars from {a.set}: {', '.join(n for n, _ in ex_info)}")

    text = torch.from_numpy(embed_names(a.name))
    for i, name in enumerate(a.name):
        results = sample(model, text[i : i + 1], exemplars[None], meta["stroke_bucket"],
                         a.num, a.temperature, a.top_p, device, gen, a.cfg)
        row, slug = [], name.strip().replace(" ", "-")
        for s, toks in enumerate(results):
            if toks is None:
                print(f"[gen] {name!r} sample {s}: did not finish within {MAX_TOKENS} tokens")
                continue
            svg = icon_to_svg(decode(toks), sw)
            path = out / f"{slug}_{s}.svg"
            path.write_text(svg)
            row.append(png(svg, 64))
            print(f"[gen] {path}  ({len(toks)} tokens)")
        rows.append(row)

    cols = max(len(r) for r in rows)
    sheet = Image.new("RGBA", (cols * 72, len(rows) * 72), "white")
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            sheet.alpha_composite(im, (c * 72 + 4, r * 72 + 4))
    sheet.save(out / "sheet.png")
    print(f"[gen] contact sheet (row 1 = style exemplars): {out / 'sheet.png'}")


if __name__ == "__main__":
    main()
