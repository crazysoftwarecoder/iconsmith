"""Parse an SVG icon into quantised, grid-aligned path subpaths.

Every drawable shape is flattened to M/L/C segments, transforms are applied,
coordinates are normalised to the viewBox and snapped to ``GRID``. Each shape yields
a fill subpath and/or a stroke subpath depending on its resolved ``fill``/``stroke``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import svgelements as se

from iconsmith.tokenizer import GRID, Command, Icon, Point, Subpath

_NONE_PAINTS = {None, "", "none", "transparent"}


class SvgParseError(ValueError):
    """Raised when an SVG yields no drawable subpaths."""


@dataclass
class ParsedIcon:
    icon: Icon
    stroke_width_rel: float | None
    n_dropped_elements: int


def _paint_is_set(v: object) -> bool:
    return str(v).strip().lower() not in _NONE_PAINTS if v is not None else False


def _float(v: object, default: float) -> float:
    if v is None:
        return default
    s = str(v).strip().lower()
    for unit in ("px", "pt", "em", "rem", "%"):
        s = s.removesuffix(unit)
    try:
        return float(s)
    except ValueError:
        return default


def _frame(svg: se.SVG) -> tuple[float, float, float, float]:
    """The coordinate frame svgelements reified into: (0, 0, width, height).

    With ``reify=True`` svgelements applies the viewBox transform itself, so element
    coordinates already live in the ``width`` x ``height`` space (which defaults to the
    viewBox size when width/height are absent or percentages).
    """
    w, h = _float(svg.width, 0.0), _float(svg.height, 0.0)
    if w > 0 and h > 0:
        return 0.0, 0.0, w, h
    vb = svg.viewbox
    if vb is not None and vb.width and vb.height:
        return float(vb.x or 0), float(vb.y or 0), float(vb.width), float(vb.height)
    raise SvgParseError("no usable width/height or viewBox")


def _quad_to_cubic(seg: se.QuadraticBezier) -> se.CubicBezier:
    p0, q, p1 = seg.start, seg.control, seg.end
    c1 = p0 + (q - p0) * (2.0 / 3.0)
    c2 = p1 + (q - p1) * (2.0 / 3.0)
    return se.CubicBezier(p0, c1, c2, p1)


def _raw_subpaths(path: se.Path) -> list[tuple[list[tuple[str, list[se.Point]]], bool]]:
    """Split a reified path into (commands-in-float-coords, closed) per subpath."""
    out: list[tuple[list[tuple[str, list[se.Point]]], bool]] = []
    cur: list[tuple[str, list[se.Point]]] = []
    closed = False
    for seg in path:
        if isinstance(seg, se.Move):
            if cur:
                out.append((cur, closed))
            cur, closed = [("M", [seg.end])], False
        elif isinstance(seg, se.Line):
            cur.append(("L", [seg.end]))
        elif isinstance(seg, se.CubicBezier):
            cur.append(("C", [seg.control1, seg.control2, seg.end]))
        elif isinstance(seg, se.QuadraticBezier):
            c = _quad_to_cubic(seg)
            cur.append(("C", [c.control1, c.control2, c.end]))
        elif isinstance(seg, se.Arc):
            for c in seg.as_cubic_curves():
                cur.append(("C", [c.control1, c.control2, c.end]))
        elif isinstance(seg, se.Close):
            closed = True
    if cur:
        out.append((cur, closed))
    return [sp for sp in out if sp[0] and sp[0][0][0] == "M"]


def _quantise(pt: se.Point, vb: tuple[float, float, float, float], grid: int) -> Point:
    x0, y0, w, h = vb
    scale = (grid - 1) / max(w, h)
    x = round((float(pt.x) - x0) * scale)
    y = round((float(pt.y) - y0) * scale)
    return (min(max(x, 0), grid - 1), min(max(y, 0), grid - 1))


def _build_subpath(
    raw: list[tuple[str, list[se.Point]]],
    closed: bool,
    mode: str,
    vb: tuple[float, float, float, float],
    grid: int,
) -> Subpath | None:
    cmds: list[Command] = [(c, tuple(_quantise(p, vb, grid) for p in pts)) for c, pts in raw]
    distinct = {p for _, pts in cmds for p in pts}
    if mode != "stroke":
        if len(distinct) < 3:  # a fill with no area
            return None
    elif len(distinct) == 1:  # zero-length stroke: a dot when caps are round
        cmds = [cmds[0], ("L", cmds[0][1])]
    if mode == "stroke" and closed:
        start = cmds[0][1][0]
        if cmds[-1][1][-1] != start:
            cmds.append(("L", (start,)))
    cmds.append(("Z", ()))
    return Subpath(mode, tuple(cmds))


def parse_svg(text: str, grid: int = GRID) -> ParsedIcon:
    try:
        svg = se.SVG.parse(io.StringIO(text), reify=True)
    except Exception as e:  # svgelements raises assorted exceptions on bad input
        raise SvgParseError(f"svgelements failed: {e}") from e
    vb = _frame(svg)
    subpaths: list[Subpath] = []
    stroke_widths: list[float] = []
    dropped = 0
    for el in svg.elements():
        if not isinstance(el, se.Shape):
            continue
        try:
            path = se.Path(el)
            path.reify()
        except Exception:
            dropped += 1
            continue
        vals = getattr(el, "values", {}) or {}
        if _float(vals.get("opacity"), 1.0) <= 0:  # e.g. Eva's invisible full-size rect
            dropped += 1
            continue
        filled = (_paint_is_set(vals.get("fill", "black"))
                  and _float(vals.get("fill-opacity"), 1.0) > 0)
        sw = _float(vals.get("stroke-width", 1), 1.0)
        stroked = _paint_is_set(vals.get("stroke")) and sw > 0
        if not filled and not stroked:
            dropped += 1
            continue
        fill_mode = "fill_eo" if str(vals.get("fill-rule", "")).lower() == "evenodd" else "fill_nz"
        raws = _raw_subpaths(path)
        if not raws:
            dropped += 1
            continue
        for mode in ([fill_mode] if filled else []) + (["stroke"] if stroked else []):
            for raw, closed in raws:
                sp = _build_subpath(raw, closed, mode, vb, grid)
                if sp is None:
                    dropped += 1
                else:
                    subpaths.append(sp)
        if stroked:
            stroke_widths.append(sw / max(vb[2], vb[3]))
    if not subpaths:
        raise SvgParseError("no drawable subpaths")
    swr = sum(stroke_widths) / len(stroke_widths) if stroke_widths else None
    return ParsedIcon(Icon(tuple(subpaths)), swr, dropped)
