import numpy as np

from iconsmith.text_embed import embed_names


def test_embed_names_shape_norm_and_semantics():
    e = embed_names(["arrow-up-right", "arrow_up_right", "house"])
    assert e.shape == (3, 384) and e.dtype == np.float32
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)
    assert e[0] @ e[1] > 0.99  # hyphen/underscore normalised to spaces
    assert e[0] @ e[1] > e[0] @ e[2]
