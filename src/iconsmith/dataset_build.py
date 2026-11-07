"""Build the processed dataset: one .npz per set, plus sets.json and stats.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from iconsmith import sources
from iconsmith.svg_parse import SvgParseError, parse_svg
from iconsmith.text_embed import TEXT_DIM, embed_names
from iconsmith.tokenizer import MAX_TOKENS, encode

STROKE_BUCKETS = [0.0, 0.03, 0.05, 0.07, 0.09, 0.12, 1.0]
N_STROKE_BUCKETS = 6


def stroke_bucket(stroke_width_rel: float | None) -> int:
    if stroke_width_rel is None:
        return 0
    return int(np.clip(np.digitize(stroke_width_rel, STROKE_BUCKETS) - 1, 0, N_STROKE_BUCKETS - 1))


def set_file_name(set_name: str) -> str:
    return set_name.replace("/", "__") + ".npz"


def build_dataset(
    raw_root: Path,
    out_root: Path,
    *,
    max_tokens: int = MAX_TOKENS,
    in_set_test_frac: float = 0.05,
    seed: int = 0,
    embed: bool = True,
) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    stats: dict = {}
    sets_meta: list[dict] = []
    totals = dict(n_files=0, n_parsed=0, n_too_long=0, n_empty=0, n_parse_error=0, n_tokens=0)

    for src, spec in sources.all_sets():
        files = list(sources.iter_set_files(raw_root, src, spec))
        st = dict(n_files=len(files), n_parsed=0, n_too_long=0, n_empty=0, n_parse_error=0)
        names: list[str] = []
        token_lists: list[list[int]] = []
        widths: list[float] = []
        for name, path in tqdm(files, desc=spec.name, leave=False):
            try:
                parsed = parse_svg(path.read_text(encoding="utf-8"))
            except SvgParseError:
                st["n_empty"] += 1
                continue
            except Exception:  # noqa: BLE001 - any other failure is a parse error, counted
                st["n_parse_error"] += 1
                continue
            toks = encode(parsed.icon)
            if len(toks) > max_tokens:
                st["n_too_long"] += 1
                continue
            names.append(name)
            token_lists.append(toks)
            if parsed.stroke_width_rel is not None:
                widths.append(parsed.stroke_width_rel)
        st["n_parsed"] = len(names)
        lens = np.array([len(t) for t in token_lists]) if token_lists else np.array([0])
        st["token_len_p50"] = float(np.percentile(lens, 50))
        st["token_len_p95"] = float(np.percentile(lens, 95))
        stats[spec.name] = st
        for k in ("n_files", "n_parsed", "n_too_long", "n_empty", "n_parse_error"):
            totals[k] += st[k]
        totals["n_tokens"] += int(lens.sum()) if token_lists else 0

        n = len(names)
        offsets = np.zeros(n + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lens) if token_lists else 0
        tokens = (
            np.concatenate([np.asarray(t, dtype=np.int16) for t in token_lists])
            if token_lists
            else np.zeros(0, dtype=np.int16)
        )
        split = np.zeros(n, dtype=np.int8)
        split[rng.permutation(n)[: round(n * in_set_test_frac)]] = 1
        text_emb = embed_names(names) if embed else np.zeros((n, TEXT_DIM), dtype=np.float32)
        np.savez(
            out_root / set_file_name(spec.name),
            tokens=tokens,
            offsets=offsets,
            names=np.array(names, dtype=str),
            text_emb=text_emb,
            split=split,
        )
        median_w = float(np.median(widths)) if widths else None
        sets_meta.append(
            dict(
                name=spec.name,
                library=src.library,
                render=spec.render,
                heldout=spec.heldout,
                n_icons=n,
                stroke_width_rel_median=median_w,
                stroke_bucket=stroke_bucket(median_w) if spec.render == "stroke" else 0,
            )
        )

    stats["_totals"] = totals
    (out_root / "sets.json").write_text(json.dumps(sets_meta, indent=1))
    (out_root / "stats.json").write_text(json.dumps(stats, indent=1))
    return json.loads(json.dumps(stats))


if __name__ == "__main__":
    raw = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed")
    s = build_dataset(raw, out)
    print(json.dumps(s["_totals"], indent=1))
