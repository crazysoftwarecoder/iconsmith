"""Path tokenizer: Icon <-> token ids, plus a grammar state machine for constrained decoding.

Layout of an encoded icon: ``BOS (MODE M x y (L x y | C x y x y x y)* [Z])* EOS``.
Coordinates are interleaved scalar tokens on a ``GRID``-sized grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID = 256
MAX_TOKENS = 1024

PAD, BOS, EOS, M, L, C, Z, STROKE, FILL_NZ, FILL_EO = range(10)
COORD_OFFSET = 10
VOCAB_SIZE = COORD_OFFSET + GRID

MODES = ("stroke", "fill_nz", "fill_eo")
MODE_TO_TOKEN = {"stroke": STROKE, "fill_nz": FILL_NZ, "fill_eo": FILL_EO}
TOKEN_TO_MODE = {v: k for k, v in MODE_TO_TOKEN.items()}
CMD_TO_TOKEN = {"M": M, "L": L, "C": C, "Z": Z}
TOKEN_TO_CMD = {v: k for k, v in CMD_TO_TOKEN.items()}
CMD_POINTS = {"M": 1, "L": 1, "C": 3, "Z": 0}

Point = tuple[int, int]
Command = tuple[str, tuple[Point, ...]]


@dataclass(frozen=True)
class Subpath:
    mode: str
    commands: tuple[Command, ...]


@dataclass(frozen=True)
class Icon:
    subpaths: tuple[Subpath, ...]


def _coord_token(v: int) -> int:
    if not 0 <= v < GRID:
        raise ValueError(f"coordinate {v} outside grid [0, {GRID})")
    return COORD_OFFSET + int(v)


def encode(icon: Icon) -> list[int]:
    toks = [BOS]
    for sp in icon.subpaths:
        if sp.mode not in MODE_TO_TOKEN:
            raise ValueError(f"unknown mode {sp.mode!r}")
        toks.append(MODE_TO_TOKEN[sp.mode])
        for i, (cmd, pts) in enumerate(sp.commands):
            if cmd not in CMD_TO_TOKEN:
                raise ValueError(f"unknown command {cmd!r}")
            if i == 0 and cmd != "M":
                raise ValueError("subpath must start with M")
            if i > 0 and cmd == "M":
                raise ValueError("M only allowed at subpath start")
            if len(pts) != CMD_POINTS[cmd]:
                raise ValueError(f"{cmd} expects {CMD_POINTS[cmd]} points, got {len(pts)}")
            toks.append(CMD_TO_TOKEN[cmd])
            for x, y in pts:
                toks.append(_coord_token(x))
                toks.append(_coord_token(y))
    toks.append(EOS)
    return toks


# Grammar states
_S_START, _S_AFTER_BOS, _S_AFTER_MODE, _S_COORDS, _S_AFTER_POINT, _S_AFTER_Z, _S_DONE = range(7)
_MODE_TOKENS = (STROKE, FILL_NZ, FILL_EO)


class GrammarState:
    """Tracks which token may come next. ``step`` raises ValueError on a disallowed token."""

    def __init__(self) -> None:
        self._state = _S_START
        self._coords_left = 0

    @property
    def done(self) -> bool:
        return self._state == _S_DONE

    def allowed(self) -> np.ndarray:
        a = np.zeros(VOCAB_SIZE, dtype=bool)
        s = self._state
        if s == _S_START:
            a[BOS] = True
        elif s == _S_AFTER_BOS:
            a[list(_MODE_TOKENS)] = True
        elif s == _S_AFTER_MODE:
            a[M] = True
        elif s == _S_COORDS:
            a[COORD_OFFSET:] = True
        elif s == _S_AFTER_POINT:
            a[[L, C, Z]] = True
        elif s == _S_AFTER_Z:
            a[list(_MODE_TOKENS)] = True
            a[EOS] = True
        return a

    def step(self, token: int) -> None:
        if not self.allowed()[token]:
            raise ValueError(f"token {token} not allowed in state {self._state}")
        s = self._state
        if s == _S_START:
            self._state = _S_AFTER_BOS
        elif s in (_S_AFTER_BOS, _S_AFTER_Z) and token in _MODE_TOKENS:
            self._state = _S_AFTER_MODE
        elif s == _S_AFTER_Z and token == EOS:
            self._state = _S_DONE
        elif s == _S_AFTER_MODE:
            self._coords_left = 2
            self._state = _S_COORDS
        elif s == _S_COORDS:
            self._coords_left -= 1
            if self._coords_left == 0:
                self._state = _S_AFTER_POINT
        elif s == _S_AFTER_POINT:
            if token == Z:
                self._state = _S_AFTER_Z
            else:
                self._coords_left = 2 * CMD_POINTS[TOKEN_TO_CMD[token]]
                self._state = _S_COORDS


def decode(tokens: list[int]) -> Icon:
    g = GrammarState()
    subpaths: list[Subpath] = []
    mode: str | None = None
    commands: list[Command] = []
    cmd: str | None = None
    coords: list[int] = []

    def flush_cmd() -> None:
        nonlocal cmd, coords
        if cmd is not None:
            pts = tuple((coords[i] - COORD_OFFSET, coords[i + 1] - COORD_OFFSET)
                        for i in range(0, len(coords), 2))
            commands.append((cmd, pts))
        cmd, coords = None, []

    for t in tokens:
        g.step(t)
        if t in TOKEN_TO_MODE:
            flush_cmd()
            if mode is not None:
                subpaths.append(Subpath(mode, tuple(commands)))
            mode, commands = TOKEN_TO_MODE[t], []
        elif t in TOKEN_TO_CMD:
            flush_cmd()
            cmd = TOKEN_TO_CMD[t]
        elif t >= COORD_OFFSET:
            coords.append(t)
        elif t == EOS:
            flush_cmd()
            if mode is not None:
                subpaths.append(Subpath(mode, tuple(commands)))
    if not g.done:
        raise ValueError("token sequence did not end in EOS")
    return Icon(tuple(subpaths))


def canonical_start(sp: Subpath) -> Subpath:
    """Start a subpath at its top-left vertex (min y, then min x) where rendering allows.

    The start vertex is otherwise arbitrary, and since x and y are separate tokens a sampler can
    take x from one plausible start and y from another, landing on a point that is not on the
    icon. Closed subpaths (last vertex == first) are rotated; open strokes may be reversed. Open
    fills are left alone: reversing them flips the winding and would change nonzero holes.
    """
    cmds = list(sp.commands)
    if len(cmds) < 3 or cmds[0][0] != "M" or cmds[-1][0] != "Z":
        return sp
    segs = cmds[1:-1]  # (cmd, pts) each ending at pts[-1]
    verts = [cmds[0][1][0]] + [pts[-1] for _, pts in segs]

    def yx(p: Point) -> tuple[int, int]:
        return (p[1], p[0])

    if verts[-1] == verts[0] and len(segs) > 1:  # closed: rotate
        j = min(range(len(segs)), key=lambda i: yx(verts[i]))
        if j == 0:
            return sp
        new = [("M", (verts[j],))] + segs[j:] + segs[:j] + [("Z", ())]
        return Subpath(sp.mode, tuple(new))
    if sp.mode == "stroke" and yx(verts[-1]) < yx(verts[0]):  # open stroke: reverse
        rev = []
        for i in range(len(segs) - 1, -1, -1):
            cmd, pts = segs[i]
            prev = verts[i]
            if cmd == "C":
                rev.append(("C", (pts[1], pts[0], prev)))
            else:
                rev.append((cmd, (prev,)))
        return Subpath(sp.mode, tuple([("M", (verts[-1],))] + rev + [("Z", ())]))
    return sp


def canonical_order(icon: Icon, starts: bool = False) -> Icon:
    """Sort subpaths top-to-bottom, then left-to-right, within each run of same-mode subpaths.

    Source libraries draw parts in arbitrary order, which makes the first tokens (and so the whole
    layout) needlessly hard to predict. Sorting only inside same-mode runs keeps the renderer's
    merged ``<path>`` groups, and therefore fill-rule holes, exactly as they were. With
    ``starts`` each subpath is first normalised by ``canonical_start`` and sorted by its start.
    """
    def key(sp: Subpath) -> tuple[int, int]:
        if starts:
            x, y = sp.commands[0][1][0]
            return (y, x)
        pts = [p for _, ps in sp.commands for p in ps]
        return (min(y for _, y in pts), min(x for x, _ in pts)) if pts else (0, 0)

    subpaths = [canonical_start(sp) for sp in icon.subpaths] if starts else list(icon.subpaths)
    out: list[Subpath] = []
    run: list[Subpath] = []
    for sp in subpaths:
        if run and sp.mode != run[0].mode:
            out.extend(sorted(run, key=key))
            run = []
        run.append(sp)
    out.extend(sorted(run, key=key))
    return Icon(tuple(out))
