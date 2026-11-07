# iconsmith Foundation (Milestones 1–2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data pipeline, tokenizer, model, and overfit-capable trainer for a set-conditioned icon completion transformer, ending with a measured tokens/sec on the M5 Pro and a 100-icon overfit run.

**Architecture:** SVG icons from permissively licensed sets are flattened to cubic/line paths, quantized to a 128-grid, and tokenized as `BOS (MODE M ... Z)* EOS` with interleaved x/y scalar tokens. A ~30M decoder-only transformer predicts those tokens conditioned on a prefix of [stroke-width bucket, MiniLM name embedding, K=8 exemplar style vectors from a small encoder]. Grammar masking makes every sample a valid path.

**Tech Stack:** Python 3.12, uv, PyTorch 2.7 (MPS), svgelements, cairosvg, sentence-transformers, numpy, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-07-iconsmith-design.md`

This plan covers spec milestones 1 and 2. Milestones 3–5 (full training run, eval harness, demo/README/weights) get a second plan once tokens/sec and dataset stats are known.

## Global Constraints

- Everything runs on this Mac: device is `mps` when available, else `cpu`. No CUDA code paths.
- Python 3.12, managed by `uv`. Package name `iconsmith`, layout `src/iconsmith/`.
- Grid: 128×128 (`GRID = 128`). Max icon sequence: 1024 tokens (`MAX_TOKENS = 1024`).
- Vocab: `PAD, BOS, EOS, M, L, C, Z, STROKE, FILL_NZ, FILL_EO` then 128 coordinate tokens.
- Exemplars per icon: `K = 8`. Conditioning dropout: style p=0.1, text p=0.1.
- Text encoder: frozen `sentence-transformers/all-MiniLM-L6-v2`, 384-d, run only in preprocessing.
- Sources: MIT/ISC/Apache only. Feather excluded. Material Symbols at weight 400 only (outlined/rounded/sharp). Held-out libraries: Iconoir (both styles), Bootstrap Icons, Octicons.
- `data/` and `runs/` are gitignored. Dataset build is one command.
- Commit after every task. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

Inputs the spec implies but no single feature test covers. Each has a pinned test in the task named.

1. SVGs using `<g transform>`, `<use>`, `<defs>`/`<symbol>`, or `<clipPath>` (common in Fluent, Material). Transforms must be applied; `use`/`defs`-only icons are parsed if svgelements resolves them, else skipped and counted, never crash. → Task 3.
2. Coordinates that land outside the viewBox (Phosphor, Remix overshoots) must be clamped to `[0, GRID-1]` after quantization, not wrap or raise. → Task 3.
3. An element with both `fill` and `stroke` (a filled dot in an outline set) must produce a FILL subpath and a STROKE subpath, not one or the other silently. → Task 3.
4. Icons that tokenize to more than 1024 tokens, or to zero subpaths after dropping degenerate ones, are skipped and counted in `stats.json`, never written and never raise. → Task 6.
5. A set with fewer than K+1 icons must still yield exemplars (sample with replacement, excluding the target). → Task 8.

---

### Task 1: Repo scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `src/iconsmith/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: importable package `iconsmith`; `uv run pytest` and `uv run ruff check` work.

- [ ] **Step 1: Initialise git and uv project**

```bash
cd /Users/ashwanthfernando/code/cool-project
git init -b main
uv init --package --name iconsmith --python 3.12 .
```

- [ ] **Step 2: Set dependencies in `pyproject.toml`**

Runtime: `torch>=2.7`, `numpy`, `svgelements`, `cairosvg`, `sentence-transformers`, `pillow`, `tqdm`, `requests`. Dev group: `pytest`, `ruff`. Add `[tool.ruff] line-length = 100`, `[tool.pytest.ini_options] testpaths = ["tests"]`.

- [ ] **Step 3: Write `.gitignore`**

Contents: `data/`, `runs/`, `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`, `*.png` under `docs/figures/` excluded explicitly with `!docs/figures/*.png`.

- [ ] **Step 4: Write the smoke test**

```python
# tests/test_smoke.py
def test_import():
    import iconsmith
    assert iconsmith.__version__
```

- [ ] **Step 5: Add `__version__ = "0.1.0"` to `src/iconsmith/__init__.py`, sync and run**

Run: `uv sync && uv run pytest -q && uv run ruff check .`
Expected: `1 passed`, ruff clean.

- [ ] **Step 6: Write a 5-line README stub** (project name, one-sentence goal, "under construction").

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: scaffold iconsmith package"
```

---

### Task 2: Tokenizer and grammar

**Files:**
- Create: `src/iconsmith/tokenizer.py`
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Produces:
  - `GRID: int = 128`, `MAX_TOKENS: int = 1024`
  - Token id constants `PAD=0, BOS=1, EOS=2, M=3, L=4, C=5, Z=6, STROKE=7, FILL_NZ=8, FILL_EO=9`, `COORD_OFFSET = 10`, `VOCAB_SIZE = COORD_OFFSET + GRID`
  - `@dataclass(frozen=True) Subpath(mode: str, commands: tuple[Command, ...])` where `mode in {"stroke", "fill_nz", "fill_eo"}` and `Command = tuple[str, tuple[tuple[int, int], ...]]` with `str in {"M","L","C","Z"}`; M and L carry 1 point, C carries 3, Z carries 0.
  - `@dataclass(frozen=True) Icon(subpaths: tuple[Subpath, ...])`
  - `encode(icon: Icon) -> list[int]` (includes BOS and EOS)
  - `decode(tokens: list[int]) -> Icon` (raises `ValueError` on ungrammatical input)
  - `class GrammarState` with `allowed() -> np.ndarray[bool, VOCAB_SIZE]`, `step(token: int) -> None`, `done: bool`. Fresh state expects BOS.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tokenizer.py
import numpy as np, pytest
from iconsmith.tokenizer import *

SQUARE = Icon((Subpath("stroke", (("M",((10,10),)),("L",((100,10),)),("L",((100,100),)),("L",((10,100),)),("Z",()))),))
CURVE = Icon((Subpath("fill_eo", (("M",((0,0),)),("C",((10,20),(30,40),(50,60))),("Z",()))),))

def test_roundtrip_square():
    assert decode(encode(SQUARE)) == SQUARE

def test_roundtrip_two_subpaths_mixed_modes():
    icon = Icon(SQUARE.subpaths + CURVE.subpaths)
    assert decode(encode(icon)) == icon

def test_encode_layout():
    toks = encode(CURVE)
    assert toks[0] == BOS and toks[-1] == EOS
    assert toks[1] == FILL_EO and toks[2] == M
    assert toks[3] == COORD_OFFSET + 0 and toks[4] == COORD_OFFSET + 0
    assert toks[5] == C and len(toks) == 1 + 1 + 1 + 2 + 1 + 6 + 1 + 1

def test_coord_out_of_range_rejected():
    with pytest.raises(ValueError):
        encode(Icon((Subpath("stroke", (("M",((128,0),)),("Z",()))),)))

def test_decode_rejects_ungrammatical():
    with pytest.raises(ValueError):
        decode([BOS, M, COORD_OFFSET, COORD_OFFSET, EOS])  # no mode token

def test_grammar_walk_matches_encode():
    toks = encode(Icon(SQUARE.subpaths + CURVE.subpaths))
    g = GrammarState()
    for t in toks:
        assert g.allowed()[t], f"token {t} disallowed"
        g.step(t)
    assert g.done

def test_grammar_initial_and_after_bos():
    g = GrammarState()
    assert g.allowed().sum() == 1 and g.allowed()[BOS]
    g.step(BOS)
    a = g.allowed()
    assert a[STROKE] and a[FILL_NZ] and a[FILL_EO] and not a[EOS] and not a[M]

def test_grammar_after_mode_requires_M():
    g = GrammarState(); g.step(BOS); g.step(STROKE)
    assert g.allowed().sum() == 1 and g.allowed()[M]

def test_grammar_after_Z_allows_mode_or_EOS_only():
    g = GrammarState()
    for t in encode(SQUARE)[:-1]:
        g.step(t)
    a = g.allowed()
    assert a[EOS] and a[STROKE] and a[FILL_NZ] and a[FILL_EO] and a.sum() == 4

def test_grammar_after_L_point_allows_L_C_Z_not_M():
    g = GrammarState()
    for t in [BOS, STROKE, M, COORD_OFFSET, COORD_OFFSET, L, COORD_OFFSET+5, COORD_OFFSET+5]:
        g.step(t)
    a = g.allowed()
    assert a[L] and a[C] and a[Z] and not a[M] and not a[EOS]

def test_grammar_coord_slots_allow_only_coords():
    g = GrammarState(); [g.step(t) for t in [BOS, STROKE, M]]
    a = g.allowed()
    assert a[COORD_OFFSET:].all() and not a[:COORD_OFFSET].any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tokenizer.py -q`
Expected: ImportError / failures.

- [ ] **Step 3: Implement `tokenizer.py`**

Grammar as a small state machine: states `START, AFTER_BOS, AFTER_MODE, AFTER_CMD_NEED_COORDS(n_remaining, cmd), AFTER_POINT, AFTER_Z, DONE`. After a completed M or L or C point: allow `L, C, Z`. After Z: allow mode tokens or EOS. `decode` runs the same machine and raises on any disallowed token. Minimal subpath (an M immediately followed by Z) is grammatical; dropping degenerate subpaths is the parser's job, not the tokenizer's.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tokenizer.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/iconsmith/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: path tokenizer with grammar state machine"
```

---

### Task 3: SVG parser and normaliser

**Files:**
- Create: `src/iconsmith/svg_parse.py`
- Test: `tests/test_svg_parse.py`, `tests/fixtures/*.svg`

**Interfaces:**
- Consumes: `Icon`, `Subpath`, `GRID` from Task 2.
- Produces:
  - `@dataclass ParsedIcon(icon: Icon, stroke_width_rel: float | None, n_dropped_elements: int)`; `stroke_width_rel` is stroke width divided by viewBox size, averaged over stroke elements, `None` if no stroke elements.
  - `parse_svg(text: str, grid: int = GRID) -> ParsedIcon`; raises `SvgParseError` (subclass of ValueError) when no drawable elements survive.

- [ ] **Step 1: Write fixtures**

Save to `tests/fixtures/`:
- `lucide_arrow.svg`: the Lucide `arrow-up-right` file verbatim (root `stroke="currentColor" stroke-width="2" fill="none"`, two `<path>` children).
- `tabler_circle_check.svg`: the Tabler filled `circle-check` verbatim (root `fill="currentColor"`, one path with two subpaths).
- `primitives.svg`: viewBox `0 0 24 24`, containing `<rect x="2" y="2" width="20" height="20"/>`, `<circle cx="12" cy="12" r="5"/>`, `<line x1="0" y1="0" x2="24" y2="24" stroke="black"/>`, `<polygon points="0,0 24,0 12,24"/>`.
- `transformed.svg`: viewBox `0 0 24 24`, `<g transform="translate(12 12) scale(0.5)"><rect x="-12" y="-12" width="24" height="24"/></g>`.
- `mixed_mode.svg`: viewBox `0 0 24 24`, root `stroke="currentColor" stroke-width="2" fill="none"`, `<path d="M4 4h16"/>` and `<circle cx="12" cy="12" r="2" fill="currentColor" stroke="none"/>`.
- `overshoot.svg`: viewBox `0 0 24 24`, `<path d="M-5 -5 L30 30 Z" fill="black"/>`.
- `evenodd.svg`: viewBox `0 0 24 24`, `<path fill-rule="evenodd" d="M2 2h20v20H2z M8 8h8v8H8z"/>`.
- `empty.svg`: viewBox `0 0 24 24`, only `<defs></defs>`.
- `quadratic_arc.svg`: viewBox `0 0 24 24`, `<path d="M2 12 Q12 2 22 12 A10 10 0 0 1 2 12 Z"/>`.
- `use_symbol.svg`: viewBox `0 0 24 24`, `<defs><symbol id="s"><rect x="0" y="0" width="10" height="10"/></symbol></defs><use href="#s"/>`.

- [ ] **Step 2: Write failing tests**

```python
# tests/test_svg_parse.py
from pathlib import Path
import pytest
from iconsmith.svg_parse import parse_svg, SvgParseError
from iconsmith.tokenizer import GRID

FX = Path(__file__).parent / "fixtures"
def load(n): return (FX / n).read_text()

def test_lucide_two_stroke_subpaths():
    p = parse_svg(load("lucide_arrow.svg"))
    assert len(p.icon.subpaths) == 2
    assert {s.mode for s in p.icon.subpaths} == {"stroke"}
    assert p.stroke_width_rel == pytest.approx(2/24, abs=1e-6)

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
    xs = [x for x, _ in pts]; ys = [y for _, y in pts]
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_svg_parse.py -q`
Expected: ImportError.

- [ ] **Step 4: Implement `svg_parse.py`**

Use `svgelements.SVG.parse(io.StringIO(text), reify=True)` and iterate `svg.elements()`. For every `Shape` (Path, Rect, Circle, Ellipse, Line, Polyline, Polygon): convert with `Path(shape)`, call `.reify()`. Determine mode from the raw inherited attribute strings `shape.values.get("fill")` and `shape.values.get("stroke")`: filled if fill not in `{None, "", "none", "transparent"}`; stroked if stroke not in the same set and `stroke-width` > 0. An element that is both emits a fill subpath then a stroke subpath. Fill rule from `shape.values.get("fill-rule") == "evenodd"`. Split a Path on each `Move` into subpaths; convert `QuadraticBezier` to `CubicBezier` (control points `p0 + 2/3(q - p0)`, `p1 + 2/3(q - p1)`) and `Arc` via `arc.as_cubic_curves()`. Normalise: subtract viewBox origin, divide by `max(vb_w, vb_h)`, multiply by `grid - 1`, round to int, clamp to `[0, grid-1]`. Drop subpaths with fewer than 2 distinct points after quantisation (count in `n_dropped_elements`). Add an explicit `Z` to filled subpaths that lack one; stroke subpaths keep `Z` only if the source had a close. Stroke width: `float(shape.values.get("stroke-width", 1)) / max(vb_w, vb_h)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_svg_parse.py -q`
Expected: all pass. If `use_symbol` raises something other than `SvgParseError`, wrap svgelements exceptions into `SvgParseError`.

- [ ] **Step 6: Commit**

```bash
git add src/iconsmith/svg_parse.py tests/test_svg_parse.py tests/fixtures
git commit -m "feat: SVG parser that flattens icons to quantised path subpaths"
```

---

### Task 4: Renderer and raster IoU

**Files:**
- Create: `src/iconsmith/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `Icon`, `GRID` (Task 2); `parse_svg` (Task 3).
- Produces:
  - `icon_to_svg(icon: Icon, stroke_width_rel: float | None, size: int = 24) -> str`: one `<path>` per subpath, coordinates scaled from grid to `size`, stroke subpaths rendered with `fill="none" stroke="#000" stroke-linecap="round" stroke-linejoin="round" stroke-width=stroke_width_rel*size`, fill subpaths with `fill="#000"` and `fill-rule` `nonzero`/`evenodd`.
  - `rasterize(svg_text: str, px: int) -> np.ndarray[bool]` shape `(px, px)`, True where alpha > 127, via `cairosvg.svg2png(output_width=px, output_height=px)` and Pillow.
  - `iou(a: np.ndarray, b: np.ndarray) -> float` (1.0 when both empty).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_render.py
import numpy as np, pytest
from pathlib import Path
from iconsmith.render import icon_to_svg, rasterize, iou
from iconsmith.svg_parse import parse_svg
from iconsmith.tokenizer import Icon, Subpath

FX = Path(__file__).parent / "fixtures"

def test_filled_square_covers_expected_area():
    sq = Icon((Subpath("fill_nz", (("M",((32,32),)),("L",((95,32),)),("L",((95,95),)),("L",((32,95),)),("Z",()))),))
    m = rasterize(icon_to_svg(sq, None), 64)
    assert m.shape == (64, 64)
    assert 0.20 < m.mean() < 0.30   # ~ (63/127)^2 = 0.246

def test_iou_identity_and_disjoint():
    a = np.zeros((8,8), bool); a[:4] = True
    b = ~a
    assert iou(a, a) == 1.0 and iou(a, b) == 0.0
    assert iou(np.zeros((2,2), bool), np.zeros((2,2), bool)) == 1.0

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_render.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement `render.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_render.py -q`
Expected: all pass. Note: the original fixture SVGs use `currentColor`; `rasterize` should pass `cairosvg` a black fill by wrapping: replace `currentColor` with `#000` before rendering.

- [ ] **Step 5: Commit**

```bash
git add src/iconsmith/render.py tests/test_render.py
git commit -m "feat: icon-to-SVG renderer and raster IoU"
```

---

### Task 5: Source table and fetcher

**Files:**
- Create: `src/iconsmith/sources.py`, `src/iconsmith/fetch.py`
- Test: `tests/test_sources.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) SetSpec(name: str, glob: str, render: str, heldout: bool, name_from_path: Callable[[Path], str])` with `render in {"stroke", "fill"}` (the set's *predominant* mode, used only for stats and the demo default).
  - `@dataclass(frozen=True) Source(library: str, kind: str, url: str, license: str, sets: tuple[SetSpec, ...])` with `kind in {"git", "npm"}`.
  - `SOURCES: tuple[Source, ...]`
  - `all_sets() -> list[tuple[Source, SetSpec]]`
  - `fetch_all(raw_root: Path) -> None` (idempotent: skips a library whose directory exists)
  - `iter_set_files(raw_root: Path, source: Source, spec: SetSpec) -> Iterator[tuple[str, Path]]` yielding `(icon_name, svg_path)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sources.py
from pathlib import Path
from iconsmith.sources import SOURCES, all_sets, iter_set_files

def test_set_names_unique():
    names = [s.name for _, s in all_sets()]
    assert len(names) == len(set(names))

def test_heldout_is_whole_library():
    for src in SOURCES:
        flags = {s.heldout for s in src.sets}
        assert len(flags) == 1, f"{src.library} mixes held-out and training sets"

def test_heldout_libraries_are_exactly():
    held = sorted({src.library for src in SOURCES if src.sets[0].heldout})
    assert held == ["bootstrap", "iconoir", "octicons"]

def test_feather_excluded_and_licenses_permissive():
    assert all(s.library != "feather" for s in SOURCES)
    assert all(s.license in {"MIT", "ISC", "Apache-2.0"} for s in SOURCES)

def test_name_from_path_examples():
    by = {s.name: s for _, s in all_sets()}
    assert by["octicons"].name_from_path(Path("icons/alert-24.svg")) == "alert"
    assert by["fluent/regular"].name_from_path(Path("icons/add_circle_24_regular.svg")) == "add-circle"
    assert by["remix/line"].name_from_path(Path("icons/Arrows/arrow-down-line.svg")) == "arrow-down"
    assert by["ionicons/outline"].name_from_path(Path("src/svg/airplane-outline.svg")) == "airplane"
    assert by["material/rounded"].name_from_path(Path("rounded/home.svg")) == "home"
    assert by["tabler/outline"].name_from_path(Path("icons/outline/arrow-up.svg")) == "arrow-up"

def test_iter_set_files_on_fake_tree(tmp_path):
    lib = tmp_path / "lucide" / "icons"; lib.mkdir(parents=True)
    (lib / "a.svg").write_text("<svg/>"); (lib / "a.json").write_text("{}")
    src = next(s for s in SOURCES if s.library == "lucide")
    got = list(iter_set_files(tmp_path, src, src.sets[0]))
    assert got == [("a", lib / "a.svg")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sources.py -q`

- [ ] **Step 3: Implement `sources.py` with this exact table**

| library | kind | url | license | sets (name → glob, render, heldout) |
|---|---|---|---|---|
| lucide | git | https://github.com/lucide-icons/lucide | ISC | `lucide` → `icons/*.svg`, stroke, no |
| tabler | git | https://github.com/tabler/tabler-icons | MIT | `tabler/outline` → `icons/outline/*.svg`, stroke; `tabler/filled` → `icons/filled/*.svg`, fill |
| phosphor | git | https://github.com/phosphor-icons/core | MIT | `phosphor/thin`, `phosphor/regular`, `phosphor/bold`, `phosphor/fill` → `assets/<w>/*.svg`, fill (Phosphor is authored as filled outlines; strip the `-thin`/`-bold`/`-fill` suffix from names) |
| material | npm | https://registry.npmjs.org/@material-symbols/svg-400/-/svg-400-0.48.3.tgz | Apache-2.0 | `material/outlined`, `material/rounded`, `material/sharp` → `<style>/*.svg` excluding `*-fill.svg`, fill |
| heroicons | git | https://github.com/tailwindlabs/heroicons | MIT | `heroicons/outline` → `optimized/24/outline/*.svg`, stroke; `heroicons/solid` → `optimized/24/solid/*.svg`, fill |
| remix | git | https://github.com/Remix-Design/RemixIcon | Apache-2.0 | `remix/line` → `icons/**/*-line.svg`, fill; `remix/fill` → `icons/**/*-fill.svg`, fill (strip `-line`/`-fill`) |
| fluent | npm | `@fluentui/svg-icons` latest tarball (resolve via registry JSON at fetch time) | MIT | `fluent/regular` → `icons/*_24_regular.svg`, fill; `fluent/filled` → `icons/*_24_filled.svg`, fill (name: strip `_24_<style>`, underscores → hyphens) |
| ionicons | git | https://github.com/ionic-team/ionicons | MIT | `ionicons/outline` → `src/svg/*-outline.svg`, stroke; `ionicons/sharp` → `src/svg/*-sharp.svg`, fill; `ionicons/filled` → `src/svg/*.svg` excluding `-outline`/`-sharp`, fill |
| iconoir | git | https://github.com/iconoir-icons/iconoir | MIT | `iconoir/regular` → `icons/regular/*.svg`, stroke, **heldout**; `iconoir/solid` → `icons/solid/*.svg`, fill, **heldout** |
| bootstrap | git | https://github.com/twbs/icons | MIT | `bootstrap` → `icons/*.svg`, fill, **heldout** |
| octicons | git | https://github.com/primer/octicons | MIT | `octicons` → `icons/*-24.svg`, fill, **heldout** (strip `-24`) |

Git fetch: `git clone --depth 1 --filter=blob:none --sparse` is unnecessary; plain `git clone --depth 1` into `raw_root/<library>`. npm fetch: download tarball with `requests`, extract with `tarfile` (members are under `package/`; strip that prefix) into `raw_root/<library>`. Use `tarfile.extractall(filter="data")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sources.py -q`

- [ ] **Step 5: Run the real fetch and record counts**

Run: `uv run python -c "from pathlib import Path; from iconsmith.fetch import fetch_all; fetch_all(Path('data/raw'))"` then
`uv run python -c "from pathlib import Path; from iconsmith.sources import all_sets, iter_set_files; [print(s.name, sum(1 for _ in iter_set_files(Path('data/raw'), src, s))) for src, s in all_sets()]"`
Expected: every set reports > 200 files; no set reports 0. Fix globs if any does.

- [ ] **Step 6: Commit**

```bash
git add src/iconsmith/sources.py src/iconsmith/fetch.py tests/test_sources.py
git commit -m "feat: icon source table and fetcher"
```

---

### Task 6: Dataset builder

**Files:**
- Create: `src/iconsmith/dataset_build.py`, `src/iconsmith/text_embed.py`, `scripts/build_dataset.sh`
- Test: `tests/test_dataset_build.py`

**Interfaces:**
- Consumes: Tasks 2, 3, 5.
- Produces:
  - `embed_names(names: list[str]) -> np.ndarray` shape `(n, 384)` float32, L2-normalised, names preprocessed by replacing `-`/`_` with spaces. Model `sentence-transformers/all-MiniLM-L6-v2`, loaded once, device `cpu`.
  - `build_dataset(raw_root: Path, out_root: Path, *, max_tokens: int = MAX_TOKENS, in_set_test_frac: float = 0.05, seed: int = 0, embed: bool = True) -> dict` returning the stats dict it also writes.
  - On-disk layout under `out_root`:
    - `<set_name with / → __>.npz` with arrays `tokens` (int16, concatenated), `offsets` (int64, len n+1), `names` (str array), `text_emb` (float32 `(n,384)`, zeros if `embed=False`), `split` (int8: 0 train, 1 in-set test).
    - `sets.json`: list of `{name, library, render, heldout, n_icons, stroke_width_rel_median, stroke_bucket}`.
    - `stats.json`: per set `{n_files, n_parsed, n_too_long, n_empty, n_parse_error, token_len_p50, token_len_p95}` and totals.
  - `STROKE_BUCKETS = [0.0, 0.03, 0.05, 0.07, 0.09, 0.12, 1.0]` → bucket index 0..5 via `np.digitize`; fill-only sets get bucket 0; `N_STROKE_BUCKETS = 6`.

- [ ] **Step 1: Write failing tests** (fixture tree built in `tmp_path`, using a monkeypatched `SOURCES` of one fake git library with two sets pointing at copied fixtures; `embed=False`)

```python
# tests/test_dataset_build.py
import json, numpy as np, pytest
from pathlib import Path
from iconsmith import sources
from iconsmith.sources import Source, SetSpec
from iconsmith.dataset_build import build_dataset, STROKE_BUCKETS
from iconsmith.tokenizer import BOS, EOS, decode

FX = Path(__file__).parent / "fixtures"

@pytest.fixture
def fake_raw(tmp_path, monkeypatch):
    lib = tmp_path / "raw" / "fake"
    (lib / "a").mkdir(parents=True); (lib / "b").mkdir()
    for i in range(12):
        (lib / "a" / f"icon-{i}.svg").write_text((FX / "lucide_arrow.svg").read_text())
        (lib / "b" / f"icon-{i}.svg").write_text((FX / "tabler_circle_check.svg").read_text())
    (lib / "a" / "broken.svg").write_text("<svg viewBox='0 0 24 24'></svg>")
    (lib / "b" / "huge.svg").write_text(
        "<svg viewBox='0 0 24 24'><path d='" + "M1 1 " + "L2 2 L1 1 " * 400 + "'/></svg>")
    src = Source("fake", "git", "x", "MIT", (
        SetSpec("fake/a", "a/*.svg", "stroke", False, lambda p: p.stem),
        SetSpec("fake/b", "b/*.svg", "fill", False, lambda p: p.stem)))
    monkeypatch.setattr(sources, "SOURCES", (src,))
    return tmp_path / "raw"

def test_build_writes_sets_and_stats(fake_raw, tmp_path):
    out = tmp_path / "processed"
    stats = build_dataset(fake_raw, out, embed=False, seed=0)
    assert (out / "fake__a.npz").exists() and (out / "fake__b.npz").exists()
    s = json.loads((out / "stats.json").read_text())
    assert s["fake/a"]["n_empty"] == 1 and s["fake/a"]["n_parsed"] == 12
    assert s["fake/b"]["n_too_long"] == 1 and s["fake/b"]["n_parsed"] == 12
    assert stats == s

def test_npz_contents_decode(fake_raw, tmp_path):
    out = tmp_path / "processed"; build_dataset(fake_raw, out, embed=False)
    z = np.load(out / "fake__a.npz", allow_pickle=False)
    n = len(z["offsets"]) - 1
    assert n == 12 and z["text_emb"].shape == (12, 384) and len(z["names"]) == 12
    for i in range(n):
        t = z["tokens"][z["offsets"][i]:z["offsets"][i+1]].tolist()
        assert t[0] == BOS and t[-1] == EOS
        decode(t)

def test_in_set_split_fraction(fake_raw, tmp_path):
    out = tmp_path / "processed"; build_dataset(fake_raw, out, embed=False, in_set_test_frac=0.25)
    z = np.load(out / "fake__a.npz")
    assert z["split"].sum() == 3

def test_sets_json_stroke_bucket(fake_raw, tmp_path):
    out = tmp_path / "processed"; build_dataset(fake_raw, out, embed=False)
    sets = {s["name"]: s for s in json.loads((out / "sets.json").read_text())}
    assert sets["fake/a"]["stroke_bucket"] == int(np.digitize(2/24, STROKE_BUCKETS)) - 1
    assert sets["fake/b"]["stroke_bucket"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement `text_embed.py` and `dataset_build.py`**

Skip (and count) an icon when `parse_svg` raises `SvgParseError` (`n_empty`), any other exception (`n_parse_error`), or `len(encode(icon)) > max_tokens` (`n_too_long`). Split: per set, `rng.random(n) < in_set_test_frac` with `np.random.default_rng(seed)`; round so a 12-icon set with frac 0.25 gives exactly 3 by using `rng.permutation(n)[:round(n*frac)]`. Progress with `tqdm`.

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Write `scripts/build_dataset.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
uv run python -m iconsmith.fetch data/raw
uv run python -m iconsmith.dataset_build data/raw data/processed
```
Give both modules an `if __name__ == "__main__":` taking positional paths.

- [ ] **Step 6: Run the full build and read the stats**

Run: `bash scripts/build_dataset.sh && uv run python -c "import json; s=json.load(open('data/processed/stats.json')); print(json.dumps(s['_totals'], indent=1)); [print(k, v['n_parsed'], v['n_too_long'], v['token_len_p95']) for k,v in s.items() if k!='_totals']"`
Expected: totals `n_parsed` > 40,000; no set with `n_too_long` > 20% of files. Record the numbers in the commit message.

- [ ] **Step 7: Commit**

```bash
git add src/iconsmith/dataset_build.py src/iconsmith/text_embed.py scripts/build_dataset.sh tests/test_dataset_build.py
git commit -m "feat: dataset builder with splits, stats and name embeddings"
```

---

### Task 7: Round-trip acceptance check and contact sheet

**Files:**
- Create: `scripts/check_roundtrip.py`, `scripts/contact_sheet.py`, `docs/figures/.gitkeep`

**Interfaces:**
- Consumes: `render`, `svg_parse`, `tokenizer`, processed data from Task 6.
- Produces: `docs/figures/roundtrip.json` and `docs/figures/contact_<set>.png`.

- [ ] **Step 1: Write `scripts/check_roundtrip.py`**

Signature `main(raw_root: Path, n: int = 500, px: int = 128, seed: int = 0) -> dict`. Samples `n` SVG files uniformly across all sets via `iter_set_files`, computes `iou(rasterize(original), rasterize(icon_to_svg(decode(encode(parsed.icon)), parsed.stroke_width_rel)))`, prints and writes `{mean, p5, p50, n, worst: [(set, name, iou) x 10]}`.

- [ ] **Step 2: Run it**

Run: `uv run python scripts/check_roundtrip.py data/raw`
Expected: `mean > 0.98` and `p5 > 0.90`. If `p5` fails and the worst cases are thin-stroke sets, set `GRID = 256` in `tokenizer.py`, re-run Tasks 2–6 tests, rebuild, and re-run. Record the outcome in the commit.

- [ ] **Step 3: Write `scripts/contact_sheet.py`**

For each set in `sets.json`, render 64 random decoded icons at 48 px into an 8×8 grid PNG (Pillow) at `docs/figures/contact_<set with / → __>.png`.

- [ ] **Step 4: Run it and eyeball three sheets** (one stroke set, one fill set, one held-out set). Icons must look like the originals.

- [ ] **Step 5: Commit**

```bash
git add scripts/check_roundtrip.py scripts/contact_sheet.py docs/figures
git commit -m "chore: round-trip acceptance check and contact sheets"
```

---

### Task 8: Torch dataset and collate

**Files:**
- Create: `src/iconsmith/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: processed `.npz` + `sets.json` layout from Task 6; `PAD`, `K` constants.
- Produces:
  - `K: int = 8`
  - `class IconDataset(torch.utils.data.Dataset)`: `__init__(self, processed_root: Path, split: str, *, k: int = K, seed: int = 0)` where `split in {"train", "inset_test", "heldout"}`: `train` = split==0 rows of non-heldout sets; `inset_test` = split==1 rows of non-heldout sets; `heldout` = all rows of heldout sets. `__getitem__(i) -> dict(tokens: LongTensor[L], text_emb: FloatTensor[384], exemplars: LongTensor[k, Lmax_k], stroke_bucket: int, set_idx: int, name: str)`. Exemplars are drawn from the same set's *train-eligible* rows for train (for test splits, from the same set's rows excluding the item), with replacement when fewer than k candidates exist, never including the item itself.
  - `collate(batch: list[dict]) -> dict` padding `tokens` to `[B, L]` and `exemplars` to `[B, k, Le]` with `PAD`, plus `text_emb [B,384]`, `stroke_bucket [B]`, `set_idx [B]`.
  - `set_names(processed_root) -> list[str]` (index order used by `set_idx`).

- [ ] **Step 1: Write `tests/conftest.py` fixtures and the failing tests**

Move the `fake_raw` fixture from Task 6 into `tests/conftest.py` and add two session-scoped fixtures there: `processed` (runs `build_dataset(fake_raw, tmp_path_factory.mktemp("p"), embed=False)` and returns the dir) and `processed_tiny` (same, but with a fake library holding a single 3-icon set). Task 6's test module keeps working unchanged.

```python
# tests/test_data.py
import torch
from iconsmith.data import IconDataset, collate, K
from iconsmith.tokenizer import PAD, BOS

def test_item_shapes(processed):            # fixture builds processed dir
    ds = IconDataset(processed, "train")
    it = ds[0]
    assert it["tokens"][0] == BOS and it["exemplars"].shape[0] == K
    assert it["text_emb"].shape == (384,)

def test_exemplars_exclude_self_and_same_set(processed):
    ds = IconDataset(processed, "train", seed=1)
    for i in range(len(ds)):
        it = ds[i]
        for ex in it["exemplars"]:
            ex = ex[ex != PAD]
            assert not torch.equal(ex, it["tokens"]) or ds.set_size(it["set_idx"]) == 1

def test_tiny_set_samples_with_replacement(processed_tiny):  # a set of 3 icons
    ds = IconDataset(processed_tiny, "train")
    it = ds[0]
    assert it["exemplars"].shape[0] == K

def test_collate_pads(processed):
    ds = IconDataset(processed, "train")
    b = collate([ds[0], ds[1]])
    assert b["tokens"].shape[0] == 2 and b["tokens"].shape[1] == max(len(ds[0]["tokens"]), len(ds[1]["tokens"]))
    assert b["exemplars"].shape[:2] == (2, K)
    assert b["stroke_bucket"].dtype == torch.long

def test_splits_disjoint(processed):
    tr = IconDataset(processed, "train"); te = IconDataset(processed, "inset_test")
    assert set(tr.keys()).isdisjoint(te.keys())    # keys() -> list of (set, name)
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement `data.py`** (add `keys()` and `set_size(set_idx)` helpers used by the tests)

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add src/iconsmith/data.py tests/test_data.py
git commit -m "feat: torch dataset with same-set exemplar sampling"
```

---

### Task 9: Model

**Files:**
- Create: `src/iconsmith/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `VOCAB_SIZE`, `PAD`, `N_STROKE_BUCKETS`, `K`.
- Produces:
  - `@dataclass ModelConfig(d_model=512, n_layers=8, n_heads=8, d_ff=1408, enc_d_model=384, enc_layers=4, enc_heads=6, max_len=1024, text_dim=384, k=K, p_drop_style=0.1, p_drop_text=0.1, dropout=0.0)`
  - `class IconSmith(nn.Module)`:
    - `__init__(cfg: ModelConfig)`
    - `encode_style(exemplars: LongTensor[B,k,Le]) -> FloatTensor[B,k,d_model]`: shared token embedding → encoder (pre-norm, RoPE, SwiGLU, SDPA with key padding mask) → masked mean-pool per exemplar → linear `enc_d_model→d_model`.
    - `forward(tokens: LongTensor[B,L], text_emb: FloatTensor[B,384], exemplars: LongTensor[B,k,Le], stroke_bucket: LongTensor[B], *, drop_style: BoolTensor[B] | None = None, drop_text: BoolTensor[B] | None = None) -> FloatTensor[B, L-1, VOCAB_SIZE]` (logits predicting `tokens[:,1:]`). Prefix = `[stroke_emb(1), text_proj(1), style(k)]` = `k+2` positions, then the icon tokens; causal mask over the whole sequence; logits returned only for icon positions. In training mode, when `drop_*` is None, draw Bernoulli masks with `p_drop_style`/`p_drop_text`; dropped style tokens are replaced by a learned `null_style` vector, dropped text by a learned `null_text` vector. In eval mode nothing is dropped unless masks are passed.
    - `loss(batch: dict) -> Tensor` = cross-entropy with `ignore_index=PAD` over `tokens[:,1:]`.
    - `n_params() -> int`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model.py
import torch
from iconsmith.model import IconSmith, ModelConfig
from iconsmith.tokenizer import VOCAB_SIZE, PAD, BOS, EOS, STROKE, M, COORD_OFFSET

def tiny(): return ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, enc_d_model=32, enc_layers=1, enc_heads=4, max_len=64)

def fake_batch(B=2, L=12, k=8, Le=10):
    tok = torch.full((B, L), PAD); tok[:, 0] = BOS; tok[:, 1] = STROKE; tok[:, 2] = M
    tok[:, 3:5] = COORD_OFFSET + 3; tok[:, 5] = EOS
    ex = torch.full((B, k, Le), PAD); ex[:, :, :6] = tok[:, None, :6]
    return dict(tokens=tok, text_emb=torch.randn(B, 384), exemplars=ex, stroke_bucket=torch.tensor([1, 0]))

def test_forward_shape():
    m = IconSmith(tiny()).eval(); b = fake_batch()
    out = m(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"])
    assert out.shape == (2, 11, VOCAB_SIZE)

def test_loss_finite_and_backward():
    m = IconSmith(tiny()).train(); b = fake_batch()
    l = m.loss(b); assert torch.isfinite(l); l.backward()
    assert m.tok_emb.weight.grad is not None

def test_causal_no_future_leak():
    m = IconSmith(tiny()).eval(); b = fake_batch()
    out1 = m(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"])
    b["tokens"][:, 5] = COORD_OFFSET + 9   # change a later token
    out2 = m(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"])
    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5)

def test_style_dropout_changes_output():
    m = IconSmith(tiny()).eval(); b = fake_batch()
    a = m(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"])
    c = m(b["tokens"], b["text_emb"], b["exemplars"], b["stroke_bucket"], drop_style=torch.tensor([True, True]))
    assert not torch.allclose(a, c)

def test_param_count_default_config():
    n = IconSmith(ModelConfig()).n_params()
    assert 25_000_000 < n < 45_000_000

def test_runs_on_mps_if_available():
    if not torch.backends.mps.is_available(): return
    m = IconSmith(tiny()).to("mps"); b = {k: (v.to("mps") if torch.is_tensor(v) else v) for k, v in fake_batch().items()}
    assert torch.isfinite(m.loss(b))
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement `model.py`**

Write RoPE, RMSNorm, SwiGLU and a `Block` once and reuse for encoder and decoder (encoder is bidirectional with a key-padding mask; decoder is causal). Use `torch.nn.functional.scaled_dot_product_attention`. Tie nothing except the token embedding shared between encoder and decoder (`tok_emb`, exposed as that attribute). Output head is a separate linear.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model.py -q`

- [ ] **Step 5: Commit**

```bash
git add src/iconsmith/model.py tests/test_model.py
git commit -m "feat: set-conditioned icon transformer"
```

---

### Task 10: Trainer with overfit mode and throughput measurement

**Files:**
- Create: `src/iconsmith/train.py`, `scripts/train.sh`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: Tasks 8, 9.
- Produces:
  - `@dataclass TrainConfig(processed: Path, run_dir: Path, epochs=20, batch_size=32, lr=3e-4, warmup_steps=500, weight_decay=0.1, grad_clip=1.0, overfit_n: int | None = None, bf16: bool = True, seed=0, log_every=20, eval_every=500, max_steps: int | None = None, device: str = "auto")`
  - `train(cfg: TrainConfig, model_cfg: ModelConfig = ModelConfig()) -> dict` returning `{final_train_loss, final_val_loss, tokens_per_sec, steps}`; writes `run_dir/log.csv` (`step,train_loss,val_loss,lr,tokens_per_sec`), `run_dir/ckpt_last.pt` (`{model_cfg, train_cfg, state_dict, step}`), `run_dir/loss.png`.
  - CLI `python -m iconsmith.train --processed data/processed --run-dir runs/x [--overfit 100] [--epochs N] [--max-steps N] [--no-bf16]`.
  - `overfit_n`: restrict train set to the first `n` items and use them as the validation set too.

- [ ] **Step 1: Write failing test** (tiny model, fixture processed dir, CPU, `max_steps=30`)

```python
# tests/test_train.py
import csv
from iconsmith.train import train, TrainConfig
from iconsmith.model import ModelConfig

def test_train_smoke_writes_artifacts(processed, tmp_path):
    cfg = TrainConfig(processed=processed, run_dir=tmp_path / "run", epochs=1, batch_size=4,
                      max_steps=30, overfit_n=8, bf16=False, device="cpu", log_every=10, eval_every=15)
    res = train(cfg, ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, enc_d_model=32, enc_layers=1, enc_heads=4, max_len=256))
    assert res["steps"] == 30 and res["tokens_per_sec"] > 0
    rows = list(csv.DictReader(open(tmp_path / "run" / "log.csv")))
    assert len(rows) >= 3 and (tmp_path / "run" / "ckpt_last.pt").exists() and (tmp_path / "run" / "loss.png").exists()
    assert float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"])
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement `train.py`**

Device: `"mps"` if available else `"cpu"` when `device="auto"`. Autocast: `torch.autocast(device_type=device, dtype=torch.bfloat16)` when `bf16`; wrap in `try` on first step and fall back to fp32 with a printed warning if MPS raises. AdamW with cosine-to-10% schedule after linear warmup. `tokens_per_sec` = non-pad icon tokens processed / wall time, averaged over the run after step 10. Validation = mean loss over the `inset_test` split (or the overfit subset) with `torch.no_grad()`. `DataLoader(num_workers=0)` on MPS.

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Overfit run on real data**

Run: `uv run python -m iconsmith.train --processed data/processed --run-dir runs/overfit100 --overfit 100 --epochs 200 --batch-size 16`
Expected: final train loss < 0.05. If loss plateaus above that, check that exemplars and targets come from the same set and that the causal mask covers the prefix correctly before anything else.

- [ ] **Step 6: Throughput run and size decision**

Run: `uv run python -m iconsmith.train --processed data/processed --run-dir runs/tput --max-steps 200 --batch-size 32`
Record `tokens_per_sec` and compute hours per epoch = (total train tokens from `stats.json`) / tokens_per_sec / 3600. Target ≤ 20 min per epoch. If over, reduce `n_layers` to 6 and/or `d_model` to 384 and re-measure; write the chosen config and the measured numbers into `docs/superpowers/plans/2026-11-07-iconsmith-foundation.md` under a new "Measured" heading at the bottom.

- [ ] **Step 7: Write `scripts/train.sh`** (full run with the chosen config, `runs/v1`) and commit

```bash
git add src/iconsmith/train.py scripts/train.sh tests/test_train.py docs/superpowers/plans
git commit -m "feat: trainer with overfit mode; record M5 Pro throughput"
```

---

## Verification (end of this plan)

- `uv run pytest -q` passes; `uv run ruff check .` clean.
- `docs/figures/roundtrip.json` shows mean > 0.98, p5 > 0.90.
- `runs/overfit100/log.csv` final train loss < 0.05.
- Measured tokens/sec and the chosen model config recorded under "Measured" below.
- Contact sheets for a stroke set, a fill set, and a held-out set look like the originals.

## Measured

(filled in by Task 10)
