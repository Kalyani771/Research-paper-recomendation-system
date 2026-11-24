"""
Sentence embeddings using Sentence-Transformers' all-MiniLM-L6-v2.

Embeddings for the paper dataset are computed once and cached to disk
(keyed by a hash of the dataset content), so the model does not need to
re-encode the whole dataset on every search or even on every restart,
only when the underlying text actually changes.
"""
import hashlib
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

_model = None  # lazy-loaded singleton; loading the model takes a moment


def get_model(model_name: str):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model '%s' (first load only)...", model_name)
        _model = SentenceTransformer(model_name)
    return _model


def _dataset_fingerprint(texts) -> str:
    joined = "\n".join(texts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def get_or_build_embeddings(texts, model_name: str, cache_dir: str) -> np.ndarray:
    """Return an (N, dim) float32 array of embeddings for `texts`, one row per
    text, in the same order. Loads from cache when the text content matches
    a previous run; otherwise encodes and writes a fresh cache file."""
    os.makedirs(cache_dir, exist_ok=True)
    fingerprint = _dataset_fingerprint(texts)
    cache_path = os.path.join(cache_dir, f"embeddings_{fingerprint}.npy")

    if os.path.exists(cache_path):
        logger.info("Loading cached paper embeddings from %s", cache_path)
        return np.load(cache_path)

    model = get_model(model_name)
    logger.info(
        "Encoding %d papers with %s (no cache hit). This is a one-time step; "
        "on a CPU expect several minutes for tens of thousands of papers...",
        len(texts), model_name,
    )
    embeddings = model.encode(
        list(texts), batch_size=64, show_progress_bar=len(texts) > 1000,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")

    np.save(cache_path, embeddings)
    logger.info("Cached embeddings to %s", cache_path)
    return embeddings


def embed_texts(texts, model_name: str) -> np.ndarray:
    """Embed a small batch on the fly (e.g. live platform results). Same
    model + normalization as the dataset, so scores are directly comparable."""
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    model = get_model(model_name)
    return model.encode(
        list(texts), batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")


def embed_query(query_text: str, model_name: str) -> np.ndarray:
    model = get_model(model_name)
    vec = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype("float32")[0]
