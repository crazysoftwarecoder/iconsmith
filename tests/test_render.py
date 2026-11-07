from pathlib import Path

import numpy as np
import pytest

from iconsmith.render import icon_to_svg, iou, rasterize
from iconsmith.svg_parse import parse_svg
from iconsmith.tokenizer import GRID, Icon, Subpath

FX = Path(__file__).parent / "fixtures"


def test_filled_square_covers_expected_area():
    sq = Icon(
        (
            Subpath(
                "fill_nz",
                (
                    ("M", ((GRID // 4, GRID // 4),)),
                    ("L", ((3 * GRID // 4, GRID // 4),)),
                    ("L", ((3 * GRID // 4, 3 * GRID // 4),)),
                    ("L", ((GRID // 4, 3 * GRID // 4),)),
                    ("Z", ()),
                ),
            ),
        )
    )
    m = rasterize(icon_to_svg(sq, None), 64)
    assert m.shape == (64, 64) and m.dtype == bool
    assert 0.22 < m.mean() < 0.28  # ~ (1/2)^2


def test_iou_identity_and_disjoint():
    a = np.zeros((8, 8), bool)
    a[:4] = True
    b = ~a
    assert iou(a, a) == 1.0 and iou(a, b) == 0.0
    assert iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 1.0


@pytest.mark.parametrize("name", ["lucide_arrow.svg", "tabler_circle_check.svg", "evenodd.svg"])
def test_parse_render_roundtrip_iou(name):
    src = (FX / name).read_text()
    p = parse_svg(src)
    orig = rasterize(src, 128)
    rt = rasterize(icon_to_svg(p.icon, p.stroke_width_rel), 128)
    assert iou(orig, rt) > 0.90


def test_evenodd_has_hole():
    p = parse_svg((FX / "evenodd.svg").read_text())
    m = rasterize(icon_to_svg(p.icon, None), 64)
    assert not m[32, 32]  # centre is a hole


def test_stroke_subpath_rendered_without_close():
    p = parse_svg((FX / "lucide_arrow.svg").read_text())
    svg = icon_to_svg(p.icon, p.stroke_width_rel)
    assert "Z" not in svg and 'fill="none"' in svg and "stroke-width" in svg


def test_dot_stroke_renders_nonempty():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
           'stroke="black" stroke-width="2" stroke-linecap="round"><path d="M8 16h.013"/></svg>')
    p = parse_svg(svg)
    m = rasterize(icon_to_svg(p.icon, p.stroke_width_rel), 64)
    assert m.sum() > 10
