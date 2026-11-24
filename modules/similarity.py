"""
Cosine similarity ranking.

Cosine similarity = (A . B) / (||A|| x ||B||)

Because embeddings.py already L2-normalizes every vector when they're
created (normalize_embeddings=True), ||A|| = ||B|| = 1, so cosine
similarity reduces to a plain dot product here. The formula above is still
exactly what's being computed.
"""
import numpy as np


def rank_papers(query_embedding: np.ndarray, paper_embeddings: np.ndarray, top_k: int = 5):
    """Return a list of (row_index, score) tuples for the top_k most similar
    papers, highest score first. Scores are cosine similarities in [-1, 1]
    (in practice close to [0, 1] for sentence embeddings)."""
    if paper_embeddings.shape[0] == 0:
        return []

    scores = paper_embeddings @ query_embedding  # dot product == cosine similarity (unit vectors)
    top_k = min(top_k, len(scores))
    top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]  # sort that small slice
    return [(int(i), float(scores[i])) for i in top_indices]
