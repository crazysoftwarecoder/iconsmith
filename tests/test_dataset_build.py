import json

import numpy as np

from iconsmith.dataset_build import STROKE_BUCKETS, build_dataset
from iconsmith.tokenizer import BOS, EOS, decode


def test_build_writes_sets_and_stats(fake_raw, tmp_path):
    out = tmp_path / "processed"
    stats = build_dataset(fake_raw, out, embed=False, seed=0)
    assert (out / "fake__a.npz").exists() and (out / "fake__b.npz").exists()
    s = json.loads((out / "stats.json").read_text())
    assert s["fake/a"]["n_empty"] == 1 and s["fake/a"]["n_parsed"] == 12
    assert s["fake/b"]["n_too_long"] == 1 and s["fake/b"]["n_parsed"] == 12
    assert stats == s


def test_npz_contents_decode(fake_raw, tmp_path):
    out = tmp_path / "processed"
    build_dataset(fake_raw, out, embed=False)
    z = np.load(out / "fake__a.npz", allow_pickle=False)
    n = len(z["offsets"]) - 1
    assert n == 12 and z["text_emb"].shape == (12, 384) and len(z["names"]) == 12
    for i in range(n):
        t = z["tokens"][z["offsets"][i] : z["offsets"][i + 1]].tolist()
        assert t[0] == BOS and t[-1] == EOS
        decode(t)


def test_in_set_split_fraction(fake_raw, tmp_path):
    out = tmp_path / "processed"
    build_dataset(fake_raw, out, embed=False, in_set_test_frac=0.25)
    z = np.load(out / "fake__a.npz")
    assert z["split"].sum() == 3


def test_sets_json_stroke_bucket(fake_raw, tmp_path):
    out = tmp_path / "processed"
    build_dataset(fake_raw, out, embed=False)
    sets = {s["name"]: s for s in json.loads((out / "sets.json").read_text())}
    assert sets["fake/a"]["stroke_bucket"] == int(np.digitize(2 / 24, STROKE_BUCKETS)) - 1
    assert sets["fake/b"]["stroke_bucket"] == 0
    assert sets["fake/a"]["heldout"] is False and sets["fake/a"]["library"] == "fake"
