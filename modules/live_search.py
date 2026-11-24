"""
Live search across existing research platforms.

Each platform's own search API is used only to *gather candidate papers*
(title + abstract + real metadata). The app then re-ranks every candidate
semantically with the same MiniLM embeddings used for the Kaggle dataset,
so the final order is decided by meaning, not by the platform's keyword
ranking.

Platforms
---------
- ieee            IEEE papers. Uses the official IEEE Xplore API when
                  IEEE_API_KEY is set; otherwise OpenAlex restricted to the
                  IEEE publisher (free, no key; real IEEE DOIs 10.1109/...).
- openalex        OpenAlex: 250M+ works from all publishers (ACM, Springer,
                  Elsevier, IEEE, ...). Free, no key.
- arxiv           arXiv preprints via the arXiv API. Free, no key.
- semanticscholar Semantic Scholar Graph API. Works without a key but is
                  heavily rate-limited; set SEMANTIC_SCHOLAR_API_KEY for
                  reliable use.

Nothing is invented: every field shown comes from the platform's response,
and a platform that fails or times out is reported as such in the UI.
"""
import hashlib
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from modules.preprocess import clean_text

logger = logging.getLogger(__name__)

USER_AGENT = "research-paper-recommender/1.0"
OPENALEX_IEEE_PUBLISHER = "P4310319808"  # "Institute of Electrical and Electronics Engineers"

PLATFORMS = {
    "ieee": "IEEE Xplore",
    "openalex": "OpenAlex",
    "arxiv": "arXiv",
    "semanticscholar": "Semantic Scholar",
}
DEFAULT_PLATFORMS = ["ieee", "openalex", "arxiv"]

_STOPWORDS = set("""
a an and are as at be based by for from has have how in into is it its of on or
that the their this to towards using use via what when which with without we
""".split())


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _content_words(query: str, limit: int = 10):
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", query)
    seen, out = set(), []
    for w in words:
        lw = w.lower()
        if lw in _STOPWORDS or lw in seen:
            continue
        seen.add(lw)
        out.append(w)
    return out[:limit]


def _openalex_abstract(inverted_index):
    if not inverted_index:
        return ""
    positions = [(pos, word) for word, poss in inverted_index.items() for pos in poss]
    return " ".join(word for _, word in sorted(positions))


def _paper(**fields):
    p = {
        "title": clean_text(fields.get("title")),
        "summary": clean_text(fields.get("summary")),
        "authors": clean_text(fields.get("authors")) or None,
        "year": fields.get("year"),
        "venue": clean_text(fields.get("venue")) or None,
        "doi": (fields.get("doi") or "").replace("https://doi.org/", "").strip() or None,
        "url": fields.get("url"),
        "platform": fields["platform"],
        "is_ieee": bool(fields.get("is_ieee")),
    }
    if not p["url"] and p["doi"]:
        p["url"] = f"https://doi.org/{p['doi']}"
    if p["doi"] and p["doi"].startswith("10.1109/"):
        p["is_ieee"] = True  # 10.1109 is IEEE's DOI prefix
    return p


def _get(url, timeout, **kw):
    headers = {"User-Agent": USER_AGENT, **kw.pop("headers", {})}
    resp = requests.get(url, timeout=timeout, headers=headers, **kw)
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------------------
# platform fetchers: each returns a list of normalized paper dicts
# --------------------------------------------------------------------------
def _openalex(query, n, timeout, ieee_only=False):
    filters = ["has_abstract:true"]
    if ieee_only:
        filters.append(f"primary_location.source.host_organization:{OPENALEX_IEEE_PUBLISHER}")
    params = {
        "filter": ",".join(filters),
        "per_page": n,
        "select": "title,publication_year,doi,primary_location,abstract_inverted_index,authorships,id",
    }
    words = _content_words(query)
    results = []
    # 1st try: all content words (precise). If that finds too little, OR them (broad).
    for search in (" ".join(words), " OR ".join(words)):
        if not search:
            break
        data = _get("https://api.openalex.org/works", timeout, params={**params, "search": search}).json()
        results = data.get("results") or []
        if len(results) >= n // 2:
            break

    papers = []
    for w in results:
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        authors = ", ".join(
            (a.get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])[:8]
        )
        papers.append(_paper(
            title=w.get("title"),
            summary=_openalex_abstract(w.get("abstract_inverted_index")),
            authors=authors,
            year=w.get("publication_year"),
            venue=src.get("display_name"),
            doi=w.get("doi"),
            url=w.get("doi") or loc.get("landing_page_url") or w.get("id"),
            platform="IEEE Xplore" if ieee_only else "OpenAlex",
            is_ieee=ieee_only,
        ))
    return papers


def _ieee_xplore(query, n, timeout, api_key):
    data = _get(
        "https://ieeexploreapi.ieee.org/api/v1/search/articles", timeout,
        params={"apikey": api_key, "querytext": query, "max_records": n, "format": "json"},
    ).json()
    papers = []
    for a in data.get("articles") or []:
        authors_block = a.get("authors") or {}
        authors = ", ".join(
            x.get("full_name", "") for x in (authors_block.get("authors") or []) if x.get("full_name")
        )
        papers.append(_paper(
            title=a.get("title"), summary=a.get("abstract"), authors=authors,
            year=a.get("publication_year"), venue=a.get("publication_title"),
            doi=a.get("doi"), url=a.get("html_url"),
            platform="IEEE Xplore", is_ieee=True,
        ))
    return papers


def _arxiv(query, n, timeout):
    words = _content_words(query)
    if not words:
        return []
    search = " OR ".join(f"all:{w}" for w in words)
    resp = _get(
        "https://export.arxiv.org/api/query", timeout,
        params={"search_query": search, "max_results": n, "sortBy": "relevance"},
    )
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(resp.content)
    papers = []
    for e in root.findall("a:entry", ns):
        published = e.findtext("a:published", default="", namespaces=ns)
        doi = e.findtext("arxiv:doi", default="", namespaces=ns)
        journal = e.findtext("arxiv:journal_ref", default="", namespaces=ns)
        authors = ", ".join(a.findtext("a:name", default="", namespaces=ns) for a in e.findall("a:author", ns)[:8])
        papers.append(_paper(
            title=e.findtext("a:title", default="", namespaces=ns),
            summary=e.findtext("a:summary", default="", namespaces=ns),
            authors=authors,
            year=int(published[:4]) if published[:4].isdigit() else None,
            venue=journal or "arXiv preprint",
            doi=doi,
            url=e.findtext("a:id", default="", namespaces=ns),
            platform="arXiv",
        ))
    return papers


def _semantic_scholar(query, n, timeout, api_key):
    headers = {"x-api-key": api_key} if api_key else {}
    data = _get(
        "https://api.semanticscholar.org/graph/v1/paper/search", timeout, headers=headers,
        params={"query": query, "limit": n, "fields": "title,abstract,year,venue,authors,externalIds,url"},
    ).json()
    papers = []
    for p in data.get("data") or []:
        authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:8])
        papers.append(_paper(
            title=p.get("title"), summary=p.get("abstract"), authors=authors,
            year=p.get("year"), venue=p.get("venue"),
            doi=(p.get("externalIds") or {}).get("DOI"), url=p.get("url"),
            platform="Semantic Scholar",
        ))
    return papers


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
_cache = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_SIZE = 64
_CACHE_TTL = 3600  # seconds

# A platform that just timed out or rate-limited us is skipped for a short
# while, so one slow/down platform doesn't add a 10 s wait to every search.
_COOLDOWN_SECONDS = 120
_cooldown_until = {}


def _start_cooldown(name):
    _cooldown_until[name] = time.time() + _COOLDOWN_SECONDS


def _norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def search_platforms(query, platforms, per_platform=20, timeout=10,
                     ieee_api_key="", semantic_scholar_api_key=""):
    """Query the chosen platforms in parallel.

    Returns (papers, status): `papers` is a de-duplicated list of
    normalized paper dicts that have an abstract; `status` maps each
    platform's display name to a short human-readable outcome.
    """
    key = (query, tuple(sorted(platforms)), per_platform)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < _CACHE_TTL:
            return hit[1], hit[2]

    jobs = {}
    if "ieee" in platforms:
        if ieee_api_key:
            jobs["IEEE Xplore"] = lambda: _ieee_xplore(query, per_platform, timeout, ieee_api_key)
        else:
            jobs["IEEE Xplore"] = lambda: _openalex(query, per_platform, timeout, ieee_only=True)
    if "openalex" in platforms:
        jobs["OpenAlex"] = lambda: _openalex(query, per_platform, timeout)
    if "arxiv" in platforms:
        jobs["arXiv"] = lambda: _arxiv(query, per_platform, timeout)
    if "semanticscholar" in platforms:
        jobs["Semantic Scholar"] = lambda: _semantic_scholar(
            query, per_platform, timeout, semantic_scholar_api_key)

    status, collected, had_error = {}, {}, False
    for name in list(jobs):
        if _cooldown_until.get(name, 0) > time.time():
            del jobs[name]
            status[name] = "temporarily unavailable (skipped)"
            had_error = True
    if jobs:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {pool.submit(fn): name for name, fn in jobs.items()}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    collected[name] = fut.result()
                    status[name] = f"{len(collected[name])} found"
                except requests.HTTPError as exc:
                    had_error = True
                    code = exc.response.status_code if exc.response is not None else "?"
                    status[name] = "rate-limited (try again later)" if code == 429 else f"failed (HTTP {code})"
                    if code == 429 or (isinstance(code, int) and code >= 500):
                        _start_cooldown(name)
                    logger.warning("%s search failed: %s", name, exc)
                except requests.Timeout:
                    had_error = True
                    status[name] = "timed out"
                    _start_cooldown(name)
                    logger.warning("%s search timed out", name)
                except Exception as exc:  # network down, bad XML/JSON, ...
                    had_error = True
                    status[name] = "unavailable"
                    _start_cooldown(name)
                    logger.warning("%s search failed: %s", name, exc)

    # Merge in a stable platform order; drop papers without an abstract
    # (nothing meaningful to embed) and duplicates across platforms.
    papers, seen = [], set()
    for name in ["IEEE Xplore", "OpenAlex", "Semantic Scholar", "arXiv"]:
        for p in collected.get(name, []):
            if not p["title"] or len(p["summary"]) < 40:
                continue
            keys = {_norm_title(p["title"])} | ({p["doi"].lower()} if p["doi"] else set())
            if keys & seen:
                continue
            seen |= keys
            p["id"] = "live-" + hashlib.sha1((p["doi"] or p["title"]).encode()).hexdigest()[:12]
            papers.append(p)

    if not had_error:  # don't cache transient failures
        with _cache_lock:
            _cache[key] = (time.time(), papers, status)
            while len(_cache) > _CACHE_SIZE:
                _cache.popitem(last=False)
    return papers, status


def normalized_title(title):
    return _norm_title(title)
