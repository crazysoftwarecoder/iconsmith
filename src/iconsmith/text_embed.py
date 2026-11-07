"""Frozen sentence embeddings for icon names (preprocessing only)."""

from __future__ import annotations

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_DIM = 384
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def normalise_name(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").strip()


def embed_names(names: list[str], batch_size: int = 256) -> np.ndarray:
    if not names:
        return np.zeros((0, TEXT_DIM), dtype=np.float32)
    model = _get_model()
    emb = model.encode(
        [normalise_name(n) for n in names],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)
