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
