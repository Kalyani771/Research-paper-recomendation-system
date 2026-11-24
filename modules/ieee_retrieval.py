"""
Live metadata retrieval and IEEE verification.

Two paths, tried in order:

1. IEEE Xplore Open API (requires config.IEEE_API_KEY). If a result comes
   back, it is genuinely IEEE by definition, since it came from IEEE's own
   catalog.
2. Crossref API (free, no key required: https://api.crossref.org). Used as
   a fallback. A Crossref match is only labeled "IEEE Xplore (verified)" if
   its own publisher/container-title field actually says IEEE; otherwise
   it's labeled "Crossref (fallback) - not confirmed as IEEE" and shown as
   such, never silently relabeled.

Nothing here invents a title, author, DOI, or link. If neither source
returns a confident match, the function says so explicitly.
"""
import difflib
import logging
import threading

import requests

logger = logging.getLogger(__name__)

SOURCE_IEEE_VERIFIED = "IEEE Xplore (verified)"
SOURCE_CROSSREF_FALLBACK = "Crossref (fallback) - not confirmed as IEEE"
SOURCE_UNVERIFIED = "Unverified - no confident match found"

_TITLE_MATCH_THRESHOLD = 0.55  # guards against returning an unrelated paper


def _title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def _query_ieee_xplore(title: str, api_key: str, timeout: int):
    try:
        resp = requests.get(
            "https://ieeexploreapi.ieee.org/api/v1/search/articles",
            params={"apikey": api_key, "querytext": title, "max_records": 1, "format": "json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("IEEE Xplore API request failed: %s", exc)
        return None
    except ValueError:
        logger.warning("IEEE Xplore API returned a non-JSON response.")
        return None

    articles = data.get("articles") or []
    if not articles:
        return None

    article = articles[0]
    found_title = article.get("title", "")
    if _title_similarity(title, found_title) < _TITLE_MATCH_THRESHOLD:
        return None

    authors_block = article.get("authors", {}) or {}
    author_list = authors_block.get("authors", []) if isinstance(authors_block, dict) else []
    authors = ", ".join(a.get("full_name", "") for a in author_list if a.get("full_name"))

    return {
        "source": SOURCE_IEEE_VERIFIED,
        "is_ieee_verified": True,
        "title": found_title or title,
        "authors": authors or None,
        "year": article.get("publication_year"),
        "venue": article.get("publication_title"),
        "doi": article.get("doi"),
        "official_url": article.get("html_url") or (
            f"https://doi.org/{article['doi']}" if article.get("doi") else None
        ),
    }


def _query_crossref(title: str, timeout: int):
    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={"query.bibliographic": title, "rows": 3},
            timeout=timeout,
            headers={"User-Agent": "research-paper-recommender/1.0 (mailto:contact@example.com)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("Crossref API request failed: %s", exc)
        return None
    except ValueError:
        logger.warning("Crossref API returned a non-JSON response.")
        return None

    items = (data.get("message") or {}).get("items") or []
    best, best_score = None, 0.0
    for item in items:
        titles = item.get("title") or []
        candidate_title = titles[0] if titles else ""
        score = _title_similarity(title, candidate_title)
        if score > best_score:
            best, best_score = item, score

    if best is None or best_score < _TITLE_MATCH_THRESHOLD:
        return None

    authors = ", ".join(
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in (best.get("author") or [])
    ).strip(", ")

    container_titles = best.get("container-title") or []
    venue = container_titles[0] if container_titles else None
    publisher = best.get("publisher", "") or ""
    is_ieee = "ieee" in publisher.lower() or "ieee" in (venue or "").lower()

    date_parts = ((best.get("issued") or {}).get("date-parts") or [[None]])[0]
    year = date_parts[0] if date_parts else None

    doi = best.get("DOI")
    return {
        "source": SOURCE_IEEE_VERIFIED if is_ieee else SOURCE_CROSSREF_FALLBACK,
        "is_ieee_verified": is_ieee,
        "title": (best.get("title") or [title])[0],
        "authors": authors or None,
        "year": year,
        "venue": venue or publisher or None,
        "doi": doi,
        "official_url": best.get("URL") or (f"https://doi.org/{doi}" if doi else None),
    }


# Successful lookups are cached (paper metadata doesn't change), so repeated
# searches don't re-hit Crossref's rate-limited public API. Failures are not
# cached, so a temporary 429 / network error gets retried next time.
_verified_cache = {}
_verified_cache_lock = threading.Lock()
_VERIFIED_CACHE_SIZE = 5000


def verify_paper(title: str, ieee_api_key: str, timeout: int = 8) -> dict:
    """Looks up real, live metadata for `title`. Returns a dict with at
    least 'source' and 'is_ieee_verified' keys; other fields may be None
    if genuinely unavailable. Never fabricates a result."""
    with _verified_cache_lock:
        if title in _verified_cache:
            return dict(_verified_cache[title])

    result = None
    if ieee_api_key:
        result = _query_ieee_xplore(title, ieee_api_key, timeout)
    if not result:
        result = _query_crossref(title, timeout)
    if result:
        with _verified_cache_lock:
            if len(_verified_cache) >= _VERIFIED_CACHE_SIZE:
                _verified_cache.pop(next(iter(_verified_cache)))
            _verified_cache[title] = result
        return dict(result)

    return {
        "source": SOURCE_UNVERIFIED,
        "is_ieee_verified": False,
        "title": title,
        "authors": None,
        "year": None,
        "venue": None,
        "doi": None,
        "official_url": None,
    }
