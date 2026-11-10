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
    fl = by["fluent/regular"].name_from_path
    assert fl(Path("icons/add_circle_24_regular.svg")) == "add-circle"
    assert by["remix/line"].name_from_path(Path("icons/Arrows/arrow-down-line.svg")) == "arrow-down"
    assert by["ionicons/outline"].name_from_path(Path("src/svg/airplane-outline.svg")) == "airplane"
    assert by["material/rounded"].name_from_path(Path("rounded/home.svg")) == "home"
    assert by["tabler/outline"].name_from_path(Path("icons/outline/arrow-up.svg")) == "arrow-up"
    assert by["phosphor/bold"].name_from_path(Path("assets/bold/acorn-bold.svg")) == "acorn"
    assert by["phosphor/regular"].name_from_path(Path("assets/regular/acorn.svg")) == "acorn"


def test_iter_set_files_on_fake_tree(tmp_path):
    lib = tmp_path / "lucide" / "icons"
    lib.mkdir(parents=True)
    (lib / "a.svg").write_text("<svg/>")
    (lib / "a.json").write_text("{}")
    src = next(s for s in SOURCES if s.library == "lucide")
    got = list(iter_set_files(tmp_path, src, src.sets[0]))
    assert got == [("a", lib / "a.svg")]


def test_iter_set_files_excludes_material_fill_and_ionicons_variants(tmp_path):
    mat = tmp_path / "material" / "rounded"
    mat.mkdir(parents=True)
    (mat / "home.svg").write_text("<svg/>")
    (mat / "home-fill.svg").write_text("<svg/>")
    src = next(s for s in SOURCES if s.library == "material")
    spec = next(s for s in src.sets if s.name == "material/rounded")
    assert [n for n, _ in iter_set_files(tmp_path, src, spec)] == ["home"]

    ion = tmp_path / "ionicons" / "src" / "svg"
    ion.mkdir(parents=True)
    for n in ["add.svg", "add-outline.svg", "add-sharp.svg"]:
        (ion / n).write_text("<svg/>")
    src = next(s for s in SOURCES if s.library == "ionicons")
    by = {s.name: s for s in src.sets}
    assert [n for n, _ in iter_set_files(tmp_path, src, by["ionicons/filled"])] == ["add"]
    assert [n for n, _ in iter_set_files(tmp_path, src, by["ionicons/outline"])] == ["add"]
    assert [n for n, _ in iter_set_files(tmp_path, src, by["ionicons/sharp"])] == ["add"]


def test_new_libraries_exclude_brands(tmp_path):
    cases = {
        "iconpark": (["source/Base/home.svg", "source/Brand/github.svg", "source/Emoji/smile.svg"],
                     "iconpark", ["home"]),
        "mingcute": (["assets/svg/core/regular/food/beef_off.svg",
                      "assets/svg/core/regular/brands/github.svg"], "mingcute/line", ["beef-off"]),
        "akar": (["src/svg/coffee.svg", "src/svg/github-fill.svg"], "akar", ["coffee"]),
        "radix": (["packages/radix-icons/icons/heart.svg",
                   "packages/radix-icons/icons/github-logo.svg"], "radix", ["heart"]),
        "carbon": (["svg/32/cafe.svg", "svg/32/4K--filled.svg", "svg/32/logo--github.svg"],
                   "carbon", ["4K-filled", "cafe"]),
        "eva": (["package/icons/outline/svg/heart-outline.svg",
                 "package/icons/outline/svg/github-outline.svg"], "eva/outline", ["heart"]),
    }
    for lib, (files, set_name, want) in cases.items():
        for f in files:
            p = tmp_path / lib / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("<svg/>")
        src = next(s for s in SOURCES if s.library == lib)
        spec = next(s for s in src.sets if s.name == set_name)
        assert [n for n, _ in iter_set_files(tmp_path, src, spec)] == want, lib


def test_iconpark_two_tone_becomes_outline():
    src = next(s for s in SOURCES if s.library == "iconpark")
    svg = '<rect fill="#2F88FF" stroke="black"/><path stroke="#FFF" d="M0 0"/>'
    out = src.fix_svg(svg)
    assert 'fill="none"' in out and "#2F88FF" not in out and 'stroke="black"' in out
    assert "#FFF" not in out
