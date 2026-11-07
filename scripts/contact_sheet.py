"""Render 64 random decoded icons per set into an 8x8 contact sheet PNG.

Usage: uv run python scripts/contact_sheet.py data/processed [docs/figures]
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import resvg_py
from PIL import Image

from iconsmith.dataset_build import set_file_name
from iconsmith.render import icon_to_svg
from iconsmith.tokenizer import decode

CELL, COLS, ROWS = 48, 8, 8


def sheet_for_set(processed: Path, meta: dict, rng: np.random.Generator) -> Image.Image:
    z = np.load(processed / set_file_name(meta["name"]), allow_pickle=False)
    n = len(z["offsets"]) - 1
    picks = rng.choice(n, size=min(COLS * ROWS, n), replace=False)
    sheet = Image.new("RGBA", (COLS * CELL, ROWS * CELL), "white")
    for j, i in enumerate(picks):
        toks = z["tokens"][z["offsets"][i] : z["offsets"][i + 1]].tolist()
        svg = icon_to_svg(decode(toks), meta["stroke_width_rel_median"], size=CELL)
        png = resvg_py.svg_to_bytes(svg_string=svg, width=CELL, height=CELL)
        im = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
        sheet.alpha_composite(im, ((j % COLS) * CELL, (j // COLS) * CELL))
    return sheet


def main(processed: Path, out_dir: Path, seed: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for meta in json.loads((processed / "sets.json").read_text()):
        if meta["n_icons"] == 0:
            continue
        path = out_dir / f"contact_{meta['name'].replace('/', '__')}.png"
        sheet_for_set(processed, meta, rng).save(path)
        print(path)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else "docs/figures"))
