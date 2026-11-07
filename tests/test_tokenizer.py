import pytest

from iconsmith.tokenizer import (
    BOS,
    COORD_OFFSET,
    EOS,
    FILL_EO,
    FILL_NZ,
    STROKE,
    C,
    GrammarState,
    Icon,
    L,
    M,
    Subpath,
    Z,
    decode,
    encode,
)

SQUARE = Icon(
    (
        Subpath(
            "stroke",
            (
                ("M", ((10, 10),)),
                ("L", ((100, 10),)),
                ("L", ((100, 100),)),
                ("L", ((10, 100),)),
                ("Z", ()),
            ),
        ),
    )
)
CURVE = Icon(
    (Subpath("fill_eo", (("M", ((0, 0),)), ("C", ((10, 20), (30, 40), (50, 60))), ("Z", ()))),)
)


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
        encode(Icon((Subpath("stroke", (("M", ((128, 0),)), ("Z", ()))),)))


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
    g = GrammarState()
    g.step(BOS)
    g.step(STROKE)
    assert g.allowed().sum() == 1 and g.allowed()[M]


def test_grammar_after_Z_allows_mode_or_EOS_only():
    g = GrammarState()
    for t in encode(SQUARE)[:-1]:
        g.step(t)
    a = g.allowed()
    assert a[EOS] and a[STROKE] and a[FILL_NZ] and a[FILL_EO] and a.sum() == 4


def test_grammar_after_L_point_allows_L_C_Z_not_M():
    g = GrammarState()
    for t in [BOS, STROKE, M, COORD_OFFSET, COORD_OFFSET, L, COORD_OFFSET + 5, COORD_OFFSET + 5]:
        g.step(t)
    a = g.allowed()
    assert a[L] and a[C] and a[Z] and not a[M] and not a[EOS]


def test_grammar_coord_slots_allow_only_coords():
    g = GrammarState()
    for t in [BOS, STROKE, M]:
        g.step(t)
    a = g.allowed()
    assert a[COORD_OFFSET:].all() and not a[:COORD_OFFSET].any()
