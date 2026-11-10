"""Declarative table of icon libraries, their licenses, and how each *set* is located.

A set is one (library, style) pair. Held-out libraries have every set held out, so the
eval measures generalisation to a library the model never saw.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SetSpec:
    name: str
    glob: str
    render: str  # predominant mode: "stroke" | "fill"
    heldout: bool
    name_from_path: Callable[[Path], str]
    exclude_suffixes: tuple[str, ...] = ()  # filename stems ending with these are skipped
    exclude_re: str | None = None  # skip files whose path (relative to the library) matches


@dataclass(frozen=True)
class Source:
    library: str
    kind: str  # "git" | "npm"
    url: str
    license: str
    sets: tuple[SetSpec, ...]
    fix_svg: Callable[[str], str] | None = None  # source-specific SVG text fix before parsing


def _stem(p: Path) -> str:
    return p.stem


def _strip(*suffixes: str) -> Callable[[Path], str]:
    def fn(p: Path) -> str:
        s = p.stem
        for suf in suffixes:
            if s.endswith(suf):
                return s[: -len(suf)]
        return s

    return fn


def _fluent_name(p: Path) -> str:
    return re.sub(r"_\d+_(regular|filled|light)$", "", p.stem).replace("_", "-")


def _underscore_name(p: Path) -> str:
    return p.stem.replace("_", "-")


def _carbon_name(p: Path) -> str:
    return p.stem.replace("--", "-")


def _iconpark_outline(svg: str) -> str:
    """IconPark sources are two-tone (blue inner fills, white inner strokes); its outline theme
    draws those as no fill and the outline colour, which is what a single-colour set should be."""
    for old, new in (('fill="#2F88FF"', 'fill="none"'), ('stroke="#FFF"', 'stroke="black"'),
                     ('stroke="white"', 'stroke="black"'), ('fill="#FFF"', 'fill="none"'),
                     ('fill="white"', 'fill="none"')):
        svg = svg.replace(old, new)
    return svg


_BRANDS = (r"(facebook|github|gitlab|google|linkedin|twitter|behance|pinterest|amazon|apple|"
           r"android|windows|figma|discord|instagram|youtube|dribbble|slack|codepen|dropbox)")

# Order matters: dataset_build draws every set's in-set test split from one RNG in this order,
# so new libraries go at the end to keep the earlier sets' splits unchanged.
SOURCES: tuple[Source, ...] = (
    Source(
        "lucide", "git", "https://github.com/lucide-icons/lucide", "ISC",
        (SetSpec("lucide", "icons/*.svg", "stroke", False, _stem),),
    ),
    Source(
        "tabler", "git", "https://github.com/tabler/tabler-icons", "MIT",
        (
            SetSpec("tabler/outline", "icons/outline/*.svg", "stroke", False, _stem),
            SetSpec("tabler/filled", "icons/filled/*.svg", "fill", False, _stem),
        ),
    ),
    Source(
        "phosphor", "git", "https://github.com/phosphor-icons/core", "MIT",
        (
            SetSpec("phosphor/thin", "assets/thin/*.svg", "fill", False, _strip("-thin")),
            SetSpec("phosphor/regular", "assets/regular/*.svg", "fill", False, _stem),
            SetSpec("phosphor/bold", "assets/bold/*.svg", "fill", False, _strip("-bold")),
            SetSpec("phosphor/fill", "assets/fill/*.svg", "fill", False, _strip("-fill")),
        ),
    ),
    Source(
        "material", "npm",
        "https://registry.npmjs.org/@material-symbols/svg-400/-/svg-400-0.48.3.tgz",
        "Apache-2.0",
        tuple(
            SetSpec(f"material/{st}", f"{st}/*.svg", "fill", False, _stem, ("-fill",))
            for st in ("outlined", "rounded", "sharp")
        ),
    ),
    Source(
        "heroicons", "git", "https://github.com/tailwindlabs/heroicons", "MIT",
        (
            SetSpec("heroicons/outline", "optimized/24/outline/*.svg", "stroke", False, _stem),
            SetSpec("heroicons/solid", "optimized/24/solid/*.svg", "fill", False, _stem),
        ),
    ),
    Source(
        "remix", "git", "https://github.com/Remix-Design/RemixIcon", "Apache-2.0",
        (
            SetSpec("remix/line", "icons/**/*-line.svg", "fill", False, _strip("-line")),
            SetSpec("remix/fill", "icons/**/*-fill.svg", "fill", False, _strip("-fill")),
        ),
    ),
    Source(
        "fluent", "npm", "https://registry.npmjs.org/@fluentui/svg-icons", "MIT",
        (
            SetSpec("fluent/regular", "icons/*_24_regular.svg", "fill", False, _fluent_name),
            SetSpec("fluent/filled", "icons/*_24_filled.svg", "fill", False, _fluent_name),
        ),
    ),
    Source(
        "ionicons", "git", "https://github.com/ionic-team/ionicons", "MIT",
        (
            SetSpec("ionicons/outline", "src/svg/*-outline.svg", "stroke", False,
                    _strip("-outline")),
            SetSpec("ionicons/sharp", "src/svg/*-sharp.svg", "fill", False, _strip("-sharp")),
            SetSpec("ionicons/filled", "src/svg/*.svg", "fill", False, _stem,
                    ("-outline", "-sharp")),
        ),
    ),
    Source(
        "iconoir", "git", "https://github.com/iconoir-icons/iconoir", "MIT",
        (
            SetSpec("iconoir/regular", "icons/regular/*.svg", "stroke", True, _stem),
            SetSpec("iconoir/solid", "icons/solid/*.svg", "fill", True, _stem),
        ),
    ),
    Source(
        "bootstrap", "git", "https://github.com/twbs/icons", "MIT",
        (SetSpec("bootstrap", "icons/*.svg", "fill", True, _stem),),
    ),
    Source(
        "octicons", "git", "https://github.com/primer/octicons", "MIT",
        (SetSpec("octicons", "icons/*-24.svg", "fill", True, _strip("-24")),),
    ),
    Source(
        "iconpark", "git", "https://github.com/bytedance/IconPark", "Apache-2.0",
        (SetSpec("iconpark", "source/*/*.svg", "stroke", False, _stem,
                 exclude_re=r"^source/(Brand|Emoji)/"),),
        fix_svg=_iconpark_outline,
    ),
    Source(
        "mingcute", "git", "https://github.com/Richard9394/MingCute", "Apache-2.0",
        tuple(
            SetSpec(f"mingcute/{st}", f"assets/svg/core/{d}/*/*.svg", "fill", False,
                    _underscore_name, exclude_re=r"/(brands|emoji|cryptocurrency)/")
            for st, d in (("line", "regular"), ("fill", "filled"))
        ),
    ),
    Source(
        "eva", "git", "https://github.com/akveo/eva-icons", "MIT",
        (
            SetSpec("eva/outline", "package/icons/outline/svg/*.svg", "fill", False,
                    _strip("-outline"), exclude_re=_BRANDS),
            SetSpec("eva/fill", "package/icons/fill/svg/*.svg", "fill", False, _stem,
                    exclude_re=_BRANDS),
        ),
    ),
    Source(
        "akar", "git", "https://github.com/artcoholic/akar-icons", "MIT",
        # the -fill icons are almost all brand logos, and fills in a stroke set anyway
        (SetSpec("akar", "src/svg/*.svg", "stroke", False, _stem, ("-fill",),
                 exclude_re=_BRANDS),),
    ),
    Source(
        "majesticons", "git", "https://github.com/halfmage/majesticons", "MIT",
        (
            SetSpec("majesticons/line", "line/*.svg", "stroke", False, _strip("-line"),
                    exclude_re=_BRANDS),
            SetSpec("majesticons/solid", "solid/*.svg", "fill", False, _stem,
                    exclude_re=_BRANDS),
        ),
    ),
    Source(
        "radix", "git", "https://github.com/radix-ui/icons", "MIT",
        (SetSpec("radix", "packages/radix-icons/icons/*.svg", "fill", False, _stem,
                 ("-logo",)),),
    ),
    Source(
        "carbon", "npm", "https://registry.npmjs.org/@carbon/icons", "Apache-2.0",
        (SetSpec("carbon", "svg/32/**/*.svg", "fill", False, _carbon_name,
                 exclude_re=r"(^|/)logo--"),),
    ),
)


def all_sets() -> list[tuple[Source, SetSpec]]:
    return [(src, s) for src in SOURCES for s in src.sets]


def iter_set_files(raw_root: Path, source: Source, spec: SetSpec) -> Iterator[tuple[str, Path]]:
    lib_root = raw_root / source.library
    for p in sorted(lib_root.glob(spec.glob)):
        if p.suffix.lower() != ".svg" or not p.is_file():
            continue
        if any(p.stem.endswith(suf) for suf in spec.exclude_suffixes):
            continue
        if spec.exclude_re and re.search(spec.exclude_re, p.relative_to(lib_root).as_posix()):
            continue
        yield spec.name_from_path(p), p
