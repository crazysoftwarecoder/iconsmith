"""Round-trip acceptance check: original SVG -> parse -> encode -> decode -> render, raster IoU.

Usage: uv run python scripts/check_roundtrip.py data/raw [n] [px]
Writes docs/figures/roundtrip.json. Acceptance: mean > 0.98 and p5 > 0.90.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from iconsmith.render import icon_to_svg, iou, rasterize
from iconsmith.sources import all_sets, iter_set_files
from iconsmith.svg_parse import SvgParseError, parse_svg
from iconsmith.tokenizer import decode, encode


def main(raw_root: Path, n: int = 500, px: int = 128, seed: int = 0) -> dict:
    rng = random.Random(seed)
    files = [
        (spec.name, name, p, src.fix_svg)
        for src, spec in all_sets()
        for name, p in iter_set_files(raw_root, src, spec)
    ]
    sample = rng.sample(files, min(n, len(files)))
    results: list[tuple[str, str, float]] = []
    skipped = 0
    for set_name, name, path, fix in sample:
        src = path.read_text(encoding="utf-8")
        src = fix(src) if fix else src  # same source-specific fix as dataset_build
        try:
            parsed = parse_svg(src)
        except SvgParseError:
            skipped += 1
            continue
        icon = decode(encode(parsed.icon))
        score = iou(rasterize(src, px), rasterize(icon_to_svg(icon, parsed.stroke_width_rel), px))
        results.append((set_name, name, score))
    scores = np.array([s for _, _, s in results])
    worst = sorted(results, key=lambda r: r[2])[:10]
    out = dict(
        n=int(len(scores)),
        skipped=skipped,
        px=px,
        mean=float(scores.mean()),
        p5=float(np.percentile(scores, 5)),
        p50=float(np.percentile(scores, 50)),
        worst=[(s, nm, round(v, 4)) for s, nm, v in worst],
        per_set={
            sn: float(np.mean([v for s, _, v in results if s == sn]))
            for sn in sorted({s for s, _, _ in results})
        },
    )
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    Path("docs/figures/roundtrip.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_set"}, indent=1))
    print("per_set:", {k: round(v, 3) for k, v in out["per_set"].items()})
    ok = out["mean"] > 0.98 and out["p5"] > 0.90
    print("ACCEPT" if ok else "REJECT")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]), *(int(a) for a in sys.argv[2:]))
