"""
Lightweight text preprocessing.

MiniLM embeddings work directly on natural sentences, so this stays light on
purpose: no stemming, no stopword removal, no lowercasing-everything. Those
steps would strip exactly the meaning we want the semantic model to use.
We only strip whitespace/control characters and reject empty input.
"""
import re

_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class EmptyQueryError(ValueError):
    pass


def clean_text(text: str) -> str:
    """Clean a single string: strip control characters and collapse whitespace.
    Preserves case, punctuation, and technical abbreviations (NLP, LLM, IoT, ...).
    """
    if text is None:
        return ""
    text = _CONTROL_CHARS_RE.sub("", str(text))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def preprocess_query(raw_query: str, min_length: int = 5) -> str:
    """Validate and clean a user-submitted query. Raises EmptyQueryError
    if the query is missing or too short to search on meaningfully."""
    cleaned = clean_text(raw_query)
    if len(cleaned) < min_length:
        raise EmptyQueryError(
            f"Please enter at least {min_length} characters describing your "
            "research title, topic, or problem statement."
        )
    return cleaned
