import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, request, jsonify

from config import Config
from modules.data_loader import load_papers
from modules.embeddings import get_or_build_embeddings, embed_query, embed_texts
from modules.similarity import rank_papers
from modules.subject_predictor import (
    train_or_load_classifier, predict_subject_area, predict_subject_areas,
)
from modules.preprocess import preprocess_query, EmptyQueryError
from modules.ieee_retrieval import verify_paper, SOURCE_IEEE_VERIFIED
from modules.live_search import (
    search_platforms, normalized_title, PLATFORMS, DEFAULT_PLATFORMS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# ---------------------------------------------------------------------------
# Startup: load dataset, build/load embeddings, train the subject classifier.
# Done once at process start so a search request never has to re-encode the
# whole dataset.
# ---------------------------------------------------------------------------
_papers_df = load_papers(Config.PAPERS_CSV_PATH)
_paper_embeddings = get_or_build_embeddings(
    _papers_df["embedding_text"].tolist(), Config.EMBEDDING_MODEL_NAME, Config.CACHE_DIR
)
_subject_clf, _subject_clf_trained = train_or_load_classifier(
    _paper_embeddings, _papers_df["subject_area"].tolist(), Config.CACHE_DIR
)

DATASET_SOURCE = "dataset"
DATASET_LABEL = "Kaggle dataset"
ALL_SOURCES = {DATASET_SOURCE: DATASET_LABEL, **PLATFORMS}
DEFAULT_SOURCES = [DATASET_SOURCE] + DEFAULT_PLATFORMS

# Live papers have no row in the dataset, so remember recent ones in memory
# for their "View Details" page (bounded, newest kept).
_live_store = OrderedDict()
_live_store_lock = threading.Lock()
_LIVE_STORE_SIZE = 2000


def _remember_live(paper):
    with _live_store_lock:
        _live_store[paper["id"]] = paper
        _live_store.move_to_end(paper["id"])
        while len(_live_store) > _LIVE_STORE_SIZE:
            _live_store.popitem(last=False)


def _form_context(**extra):
    return dict(
        example_topics=_example_topics(),
        paper_count=len(_papers_df), all_sources=ALL_SOURCES,
        default_sources=DEFAULT_SOURCES,
        ieee_key_configured=bool(Config.IEEE_API_KEY),
        s2_key_configured=bool(Config.SEMANTIC_SCHOLAR_API_KEY),
        **extra,
    )


@app.route("/")
def index():
    return render_template("index.html", **_form_context())


@app.route("/recommend", methods=["POST"])
def recommend():
    raw_query = request.form.get("query", "")
    sources = [s for s in request.form.getlist("sources") if s in ALL_SOURCES]
    if "sources_submitted" not in request.form:
        sources = sources or list(DEFAULT_SOURCES)  # e.g. a plain POST with only a query

    try:
        query = preprocess_query(raw_query, min_length=Config.MIN_QUERY_LENGTH)
        if not sources:
            raise EmptyQueryError("Please select at least one source to search.")
    except EmptyQueryError as exc:
        return render_template("index.html", **_form_context(
            error=str(exc), previous_query=raw_query, selected_sources=sources))

    query_embedding = embed_query(query, Config.EMBEDDING_MODEL_NAME)

    # Start the live platform search in the background while we rank the
    # local dataset (network wait and CPU work overlap).
    platforms = [s for s in sources if s in PLATFORMS]
    candidates = []
    platform_status = {}
    with ThreadPoolExecutor(max_workers=1) as executor:
        live_future = executor.submit(
            search_platforms, query, platforms, Config.LIVE_RESULTS_PER_PLATFORM,
            Config.LIVE_SEARCH_TIMEOUT_SECONDS, Config.IEEE_API_KEY,
            Config.SEMANTIC_SCHOLAR_API_KEY,
        ) if platforms else None

        # --- 1. Kaggle dataset: precomputed embeddings ---
        if DATASET_SOURCE in sources:
            for row_index, score in rank_papers(query_embedding, _paper_embeddings, top_k=Config.TOP_K):
                row = _papers_df.iloc[row_index]
                candidates.append({
                    "id": str(row["id"]),
                    "title": row["title"],
                    "summary": row["summary"],
                    "subject_area": row["subject_area"] or "Unclassified",
                    "score": float(score),
                    "found_via": DATASET_LABEL,
                    "live": False,
                })

        # --- 2. Live platforms: embed their abstracts on the fly ---
        if live_future is not None:
            live_papers, platform_status = live_future.result()
            if live_papers:
                emb = embed_texts([f"{p['title']}. {p['summary']}" for p in live_papers],
                                  Config.EMBEDDING_MODEL_NAME)
                scores = emb @ query_embedding
                subjects = predict_subject_areas(_subject_clf, emb)
                for p, score, subject in zip(live_papers, scores, subjects):
                    candidates.append({
                        **p,
                        "subject_area": subject or "Unclassified",
                        "score": float(score),
                        "found_via": p["platform"],
                        "live": True,
                    })

    # --- 3. merge: one semantic ranking across every source ---
    # The same paper found in the dataset AND on a platform keeps the live
    # copy on a tie, since it carries real publisher metadata.
    candidates.sort(key=lambda c: (c["score"], c["live"]), reverse=True)
    results, seen = [], set()
    for c in candidates:
        key = normalized_title(c["title"])
        if key in seen:
            continue
        seen.add(key)
        results.append(c)
        if len(results) >= Config.TOP_K:
            break

    # --- 4. metadata: live papers already have it; verify dataset papers in parallel ---
    to_verify = [r for r in results if not r["live"]]
    if to_verify:
        # At most 3 at once: Crossref's anonymous API rate-limits bursts.
        with ThreadPoolExecutor(max_workers=min(3, len(to_verify))) as executor:
            verifications = list(executor.map(
                lambda r: verify_paper(r["title"], Config.IEEE_API_KEY, Config.REQUEST_TIMEOUT_SECONDS),
                to_verify,
            ))
        for r, v in zip(to_verify, verifications):
            r["verification"] = v
    for r in results:
        if r["live"]:
            r["verification"] = _live_verification(r)
            _remember_live(r)
        r["score_percent"] = round(r["score"] * 100, 1)

    predicted_label, predicted_confidence, predicted_reliable = predict_subject_area(
        _subject_clf, query_embedding
    )

    source_status = []
    if DATASET_SOURCE in sources:
        source_status.append((DATASET_LABEL, f"{len(_papers_df):,} papers searched"))
    source_status += sorted(platform_status.items())

    return render_template(
        "results.html",
        **_form_context(selected_sources=sources, previous_query=query),
        query=query,
        results=results,
        source_status=source_status,
        predicted_label=predicted_label,
        predicted_confidence=round(predicted_confidence * 100, 1),
        predicted_reliable=predicted_reliable,
    )


def _live_verification(p):
    """Metadata for a live result comes straight from the platform that
    returned it; nothing extra is looked up or invented."""
    return {
        "source": SOURCE_IEEE_VERIFIED if p["is_ieee"] else f"{p['platform']} (live result)",
        "is_ieee_verified": p["is_ieee"],
        "title": p["title"],
        "authors": p["authors"],
        "year": p["year"],
        "venue": p["venue"],
        "doi": p["doi"],
        "official_url": p["url"],
    }


@app.route("/paper/<paper_id>")
def paper_detail(paper_id):
    if paper_id.startswith("live-"):
        with _live_store_lock:
            paper = _live_store.get(paper_id)
        if paper is None:
            return render_template(
                "error.html",
                message="This live result has expired. Please run the search again."), 404
        return render_template("paper_detail.html", paper=paper, verification=paper["verification"])

    matches = _papers_df[_papers_df["id"].astype(str) == str(paper_id)]
    if matches.empty:
        return render_template("error.html", message=f"No paper found with id '{paper_id}'."), 404

    row = matches.iloc[0]
    paper = row.to_dict()
    paper["found_via"] = DATASET_LABEL
    verification = verify_paper(row["title"], Config.IEEE_API_KEY, Config.REQUEST_TIMEOUT_SECONDS)
    return render_template("paper_detail.html", paper=paper, verification=verification)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "papers_loaded": int(len(_papers_df)),
        "embedding_model": Config.EMBEDDING_MODEL_NAME,
        "subject_classifier_trained": _subject_clf_trained,
        "ieee_api_key_configured": bool(Config.IEEE_API_KEY),
        "semantic_scholar_api_key_configured": bool(Config.SEMANTIC_SCHOLAR_API_KEY),
        "live_platforms": list(PLATFORMS.values()),
    })


def _example_topics():
    return [
        "Multilingual sentiment analysis using NLP and large language models",
        "AI-based smart parking using computer vision",
        "Intrusion detection using machine learning",
        "Edge computing for the Internet of Things",
        "Query optimization in relational database systems",
    ]


@app.errorhandler(404)
def not_found(_exc):
    return render_template("error.html", message="Page not found."), 404


@app.errorhandler(500)
def server_error(_exc):
    logger.exception("Unhandled server error")
    return render_template("error.html", message="Something went wrong processing that request."), 500


if __name__ == "__main__":
    app.run(debug=Config.DEBUG)
