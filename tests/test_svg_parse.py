from pathlib import Path

import pytest

from iconsmith.svg_parse import SvgParseError, parse_svg
from iconsmith.tokenizer import GRID

FX = Path(__file__).parent / "fixtures"


def load(n):
    return (FX / n).read_text()


def test_lucide_two_stroke_subpaths():
    p = parse_svg(load("lucide_arrow.svg"))
    assert len(p.icon.subpaths) == 2
    assert {s.mode for s in p.icon.subpaths} == {"stroke"}
    assert p.stroke_width_rel == pytest.approx(2 / 24, abs=1e-6)


def test_tabler_filled_two_subpaths_nonzero():
    p = parse_svg(load("tabler_circle_check.svg"))
    assert len(p.icon.subpaths) == 2
    assert {s.mode for s in p.icon.subpaths} == {"fill_nz"}
    assert p.stroke_width_rel is None


def test_primitives_become_paths():
    p = parse_svg(load("primitives.svg"))
    assert len(p.icon.subpaths) == 4
    cmds = [c for s in p.icon.subpaths for c, _ in s.commands]
    assert "C" in cmds  # circle became cubics
    assert all(s.commands[0][0] == "M" for s in p.icon.subpaths)


def test_transform_applied():
    p = parse_svg(load("transformed.svg"))
    pts = [pt for s in p.icon.subpaths for _, ps in s.commands for pt in ps]
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    # rect ends up covering the centre 6..18 of 24 -> grid 32..96
    assert min(xs) == pytest.approx(32, abs=1) and max(xs) == pytest.approx(96, abs=1)
    assert min(ys) == pytest.approx(32, abs=1) and max(ys) == pytest.approx(96, abs=1)


def test_mixed_mode_yields_stroke_and_fill():
    p = parse_svg(load("mixed_mode.svg"))
    assert sorted(s.mode for s in p.icon.subpaths) == ["fill_nz", "stroke"]


def test_overshoot_is_clamped():
    p = parse_svg(load("overshoot.svg"))
    for s in p.icon.subpaths:
        for _, ps in s.commands:
            for x, y in ps:
                assert 0 <= x < GRID and 0 <= y < GRID


def test_evenodd_mode():
    p = parse_svg(load("evenodd.svg"))
    assert all(s.mode == "fill_eo" for s in p.icon.subpaths)
    assert len(p.icon.subpaths) == 2


def test_empty_raises():
    with pytest.raises(SvgParseError):
        parse_svg(load("empty.svg"))


def test_quadratic_and_arc_become_cubics():
    p = parse_svg(load("quadratic_arc.svg"))
    cmds = [c for s in p.icon.subpaths for c, _ in s.commands]
    assert set(cmds) <= {"M", "L", "C", "Z"} and cmds.count("C") >= 2


def test_use_symbol_does_not_crash():
    try:
        p = parse_svg(load("use_symbol.svg"))
        assert len(p.icon.subpaths) >= 1
    except SvgParseError:
        pass  # acceptable: skipped and counted upstream


def test_all_coords_are_ints_in_grid():
    p = parse_svg(load("lucide_arrow.svg"))
    for s in p.icon.subpaths:
        for _, ps in s.commands:
            for x, y in ps:
                assert isinstance(x, int) and isinstance(y, int) and 0 <= x < GRID and 0 <= y < GRID


def test_every_subpath_ends_with_Z_and_open_stroke_has_no_return_line():
    p = parse_svg(load("lucide_arrow.svg"))
    for s in p.icon.subpaths:
        assert s.commands[-1][0] == "Z"
    # "M7 17 17 7" is an open two-point stroke: M, L, Z only
    short = min(p.icon.subpaths, key=lambda s: len(s.commands))
    assert [c for c, _ in short.commands] == ["M", "L", "Z"]


def test_closed_stroke_gets_explicit_return_line():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
           'stroke="black" stroke-width="2"><path d="M4 4h16v16H4z"/></svg>')
    p = parse_svg(svg)
    (sp,) = p.icon.subpaths
    cmds = [c for c, _ in sp.commands]
    assert cmds == ["M", "L", "L", "L", "L", "Z"]
    assert sp.commands[0][1][0] == sp.commands[-2][1][0]  # returns to start
