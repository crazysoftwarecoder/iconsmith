"""Render an ``Icon`` back to SVG text and rasterise SVG text to a boolean mask."""

from __future__ import annotations

import io

import numpy as np
import resvg_py
from PIL import Image

from iconsmith.tokenizer import GRID, Icon

_FILL_RULE = {"fill_nz": "nonzero", "fill_eo": "evenodd"}


def _path_d(commands, scale: float) -> str:
    parts = []
    for cmd, pts in commands:
        if cmd == "Z":
            continue
        coords = " ".join(f"{x * scale:.3f} {y * scale:.3f}" for x, y in pts)
        parts.append(f"{cmd} {coords}")
    return " ".join(parts)


def icon_to_svg(icon: Icon, stroke_width_rel: float | None, size: int = 24) -> str:
    """One <path> per run of consecutive same-mode subpaths, so fill rules see all holes."""
    scale = size / (GRID - 1)
    sw = (stroke_width_rel or 2 / 24) * size
    groups: list[tuple[str, list[str]]] = []
    for sp in icon.subpaths:
        d = _path_d(sp.commands, scale)
        if groups and groups[-1][0] == sp.mode:
            groups[-1][1].append(d)
        else:
            groups.append((sp.mode, [d]))
    paths = []
    for mode, ds in groups:
        if mode == "stroke":
            paths.append(
                f'<path d="{" ".join(ds)}" fill="none" stroke="#000" stroke-width="{sw:.3f}" '
                'stroke-linecap="round" stroke-linejoin="round"/>'
            )
        else:
            d = " ".join(f"{x} Z" for x in ds)
            paths.append(f'<path d="{d}" fill="#000" fill-rule="{_FILL_RULE[mode]}"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
        f'width="{size}" height="{size}">' + "".join(paths) + "</svg>"
    )


def rasterize(svg_text: str, px: int) -> np.ndarray:
    svg_text = svg_text.replace("currentColor", "#000")
    png = resvg_py.svg_to_bytes(svg_string=svg_text, width=px, height=px)
    im = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
    return np.asarray(im)[..., 3] > 127


def alpha(svg_text: str, px: int) -> np.ndarray:
    """Anti-aliased coverage in [0, 1], float32 [px, px]."""
    svg_text = svg_text.replace("currentColor", "#000")
    png = resvg_py.svg_to_bytes(svg_string=svg_text, width=px, height=px)
    im = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
    return np.asarray(im, dtype=np.float32)[..., 3] / 255.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)
