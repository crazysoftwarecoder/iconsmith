"""Download every source library into ``raw_root/<library>`` (idempotent)."""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from pathlib import Path

import requests

from iconsmith.sources import SOURCES, Source


def _resolve_tarball(url: str) -> tuple[str, str]:
    """Return (tarball_url, version). ``url`` is either a .tgz or a registry package URL."""
    if url.endswith(".tgz"):
        return url, url.rsplit("-", 1)[-1].removesuffix(".tgz")
    meta = requests.get(url, timeout=60).json()
    version = meta["dist-tags"]["latest"]
    return meta["versions"][version]["dist"]["tarball"], version


def _fetch_npm(source: Source, dest: Path) -> None:
    tarball, version = _resolve_tarball(source.url)
    data = requests.get(tarball, timeout=300).content
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = []
        for m in tf.getmembers():
            if m.name.startswith("package/"):
                m.name = m.name[len("package/"):]
                if m.name:
                    members.append(m)
        tf.extractall(dest, members=members, filter="data")
    (dest / "VERSION").write_text(version + "\n")


def _fetch_git(source: Source, dest: Path) -> None:
    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", source.url, str(dest)],
        check=True,
    )


def fetch_all(raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        dest = raw_root / src.library
        if dest.exists():
            print(f"[fetch] {src.library}: exists, skipping")
            continue
        print(f"[fetch] {src.library}: {src.kind} {src.url}")
        if src.kind == "git":
            _fetch_git(src, dest)
        elif src.kind == "npm":
            _fetch_npm(src, dest)
        else:
            raise ValueError(f"unknown source kind {src.kind}")


if __name__ == "__main__":
    fetch_all(Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw"))
