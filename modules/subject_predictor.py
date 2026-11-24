"""
Subject-area prediction.

Trains a simple scikit-learn classifier (k-nearest-neighbors over the same
MiniLM embeddings used for recommendation) on whatever subject_area labels
exist in the loaded dataset, then predicts a label for the user's query.

Honesty note: with only ~20 labeled rows in the bundled demo dataset, this
classifier is for pipeline demonstration only -- it has not been evaluated
and its accuracy should not be trusted. Train it on the full Kaggle dataset
(with real subject_area labels) before relying on its predictions; see
data/README_DATA.md.
"""
import logging
import os

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

_FALLBACK_LABEL = "Unclassified / insufficient labeled data"


def train_or_load_classifier(paper_embeddings: np.ndarray, subject_labels, cache_dir: str):
    """Returns (classifier_or_None, is_trained: bool). classifier is None
    when there isn't enough labeled, varied data to train on."""
    labels = list(subject_labels)
    non_empty = [l for l in labels if l]
    unique_labels = set(non_empty)

    if len(non_empty) < 5 or len(unique_labels) < 2:
        logger.warning(
            "Not enough labeled subject_area data (%d labeled rows, %d unique "
            "labels) to train a subject classifier. Predictions will be "
            "reported as '%s'.", len(non_empty), len(unique_labels), _FALLBACK_LABEL,
        )
        return None, False

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "subject_classifier.joblib")

    mask = np.array([bool(l) for l in labels])
    X = paper_embeddings[mask]
    y = np.array(labels)[mask]

    # Tiny datasets: k=3. Real datasets: more neighbours give smoother,
    # more meaningful confidence scores; distance weighting lets the very
    # closest papers count most.
    k = 3 if len(y) < 200 else 15
    clf = KNeighborsClassifier(n_neighbors=min(k, len(y)), metric="cosine", weights="distance")
    clf.fit(X, y)
    joblib.dump(clf, cache_path)
    logger.info("Trained subject-area classifier on %d labeled rows, %d classes.", len(y), len(set(y)))
    return clf, True


def predict_subject_areas(clf, embeddings: np.ndarray):
    """Predict a subject label for each row (used for live platform papers,
    which arrive without a subject label)."""
    if clf is None or len(embeddings) == 0:
        return [""] * len(embeddings)
    return [str(x) for x in clf.predict(embeddings)]


def predict_subject_area(clf, query_embedding: np.ndarray):
    """Returns (label, confidence_0_to_1, is_reliable). is_reliable is False
    whenever clf is None, i.e. there wasn't enough data to train on."""
    if clf is None:
        return _FALLBACK_LABEL, 0.0, False

    probs = clf.predict_proba(query_embedding.reshape(1, -1))[0]
    best_idx = int(np.argmax(probs))
    label = clf.classes_[best_idx]
    confidence = float(probs[best_idx])
    return label, confidence, True
