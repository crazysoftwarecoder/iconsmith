import torch

from iconsmith.data import IconDataset, K, collate, set_names
from iconsmith.tokenizer import BOS, PAD


def test_item_shapes(processed):
    ds = IconDataset(processed, "train")
    it = ds[0]
    assert it["tokens"][0] == BOS and it["exemplars"].shape[0] == K
    assert it["text_emb"].shape == (384,)
    assert isinstance(it["stroke_bucket"], int) and isinstance(it["set_idx"], int)
    assert isinstance(it["name"], str)


def test_exemplars_exclude_self_and_same_set(processed):
    ds = IconDataset(processed, "train", seed=1)
    for i in range(len(ds)):
        it = ds[i]
        for ex in it["exemplars"]:
            ex = ex[ex != PAD]
            assert not torch.equal(ex, it["tokens"]) or ds.set_size(it["set_idx"]) == 1


def test_tiny_set_samples_with_replacement(processed_tiny):
    ds = IconDataset(processed_tiny, "train")
    it = ds[0]
    assert it["exemplars"].shape[0] == K


def test_collate_pads(processed):
    ds = IconDataset(processed, "train")
    b = collate([ds[0], ds[1]])
    assert b["tokens"].shape[0] == 2
    assert b["tokens"].shape[1] == max(len(ds[0]["tokens"]), len(ds[1]["tokens"]))
    assert b["exemplars"].shape[:2] == (2, K)
    assert b["stroke_bucket"].dtype == torch.long and b["set_idx"].dtype == torch.long
    assert b["text_emb"].shape == (2, 384)


def test_splits_disjoint(processed):
    tr = IconDataset(processed, "train")
    te = IconDataset(processed, "inset_test")
    assert set(tr.keys()).isdisjoint(te.keys())
    assert len(tr) + len(te) == 24


def test_heldout_split_empty_when_no_heldout_sets(processed):
    assert len(IconDataset(processed, "heldout")) == 0


def test_set_names_order(processed):
    assert set_names(processed) == ["fake/a", "fake/b"]


def test_exemplars_truncated_to_max_len(processed):
    ds = IconDataset(processed, "train", exemplar_max_len=8)
    it = ds[0]
    assert it["exemplars"].shape == (K, 8)
    assert (it["exemplars"] != PAD).all()  # truncated, not padded


def test_self_content_is_perturbed_copy_of_target(processed):
    from iconsmith.tokenizer import decode

    ds = IconDataset(processed, "train", n_content=1, p_self_content=1.0, self_noise=2.0)
    for i in range(min(5, len(ds))):
        item = ds[i]
        target = decode(item["tokens"].tolist())
        c = item["content"][0]
        cont = decode(c[c != PAD].tolist())
        assert len(cont.subpaths) == len(target.subpaths)
        assert [s.mode for s in cont.subpaths] == [s.mode for s in target.subpaths]


def test_raster_content(processed):
    ds = IconDataset(processed, "train", n_content=1, content_raster=16, p_self_content=0.5)
    item = ds[0]
    assert item["content"].shape == (1, 16, 16) and item["content"].dtype == torch.float32
    assert 0 <= float(item["content"].min()) and float(item["content"].max()) <= 1
    assert float(item["content"].sum()) > 0  # something was drawn
    batch = collate([ds[0], ds[1]])
    assert batch["content"].shape == (2, 1, 16, 16)
