from pathlib import Path

import pytest

from iconsmith import sources
from iconsmith.sources import SetSpec, Source

FX = Path(__file__).parent / "fixtures"


def _make_fake_raw(root: Path, monkeypatch, tiny: bool = False) -> Path:
    lib = root / "raw" / "fake"
    (lib / "a").mkdir(parents=True)
    (lib / "b").mkdir()
    n = 3 if tiny else 12
    for i in range(n):
        (lib / "a" / f"icon-{i}.svg").write_text((FX / "lucide_arrow.svg").read_text())
        (lib / "b" / f"icon-{i}.svg").write_text((FX / "tabler_circle_check.svg").read_text())
    if not tiny:
        (lib / "a" / "broken.svg").write_text("<svg viewBox='0 0 24 24'></svg>")
        (lib / "b" / "huge.svg").write_text(
            "<svg viewBox='0 0 24 24'><path d='M1 1 " + "L2 2 L1 3 " * 400 + "'/></svg>"
        )
    sets = (SetSpec("fake/a", "a/*.svg", "stroke", False, lambda p: p.stem),)
    if not tiny:
        sets += (SetSpec("fake/b", "b/*.svg", "fill", False, lambda p: p.stem),)
    monkeypatch.setattr(sources, "SOURCES", (Source("fake", "git", "x", "MIT", sets),))
    return root / "raw"


@pytest.fixture
def fake_raw(tmp_path, monkeypatch):
    return _make_fake_raw(tmp_path, monkeypatch)


@pytest.fixture
def processed(tmp_path, monkeypatch):
    from iconsmith.dataset_build import build_dataset

    raw = _make_fake_raw(tmp_path, monkeypatch)
    out = tmp_path / "processed"
    build_dataset(raw, out, embed=False)
    return out


@pytest.fixture
def processed_tiny(tmp_path, monkeypatch):
    from iconsmith.dataset_build import build_dataset

    raw = _make_fake_raw(tmp_path, monkeypatch, tiny=True)
    out = tmp_path / "processed_tiny"
    build_dataset(raw, out, embed=False)
    return out
