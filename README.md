# Research Paper Recommendation System

A Flask app that recommends research papers by semantic meaning rather than
keyword overlap. It embeds a research title/topic/problem statement with a
MiniLM sentence-transformer, ranks candidate papers by cosine similarity,
predicts a likely subject area, and verifies each result's real metadata and
publisher live via IEEE Xplore (or Crossref as a free fallback) before
showing it.

## Honesty notes (please read before demoing this)

- **Dataset:** the app uses the Kaggle
  [arXiv Paper Abstracts](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts)
  dataset — ~41k unique arXiv papers with their real titles and abstracts.
  It is **arXiv, not IEEE**, and is heavily weighted toward computer vision
  and machine learning (it was scraped from cs.CV / cs.LG / stat.ML), so
  topics far from those areas will have fewer close matches. It has no
  year/venue/DOI columns, so that metadata comes only from the live
  verification step. See `data/README_DATA.md`.
- `subject_area` is derived from each paper's arXiv category tags (most
  specific tag wins), not hand-labelled.
- The old 20-row `data/sample_papers.csv` is kept as a tiny fallback for
  quick tests (`PAPERS_CSV_PATH=data/sample_papers.csv`).
- Nothing about IEEE verification is faked: `modules/ieee_retrieval.py` calls
  real APIs at request time and only shows metadata (DOI, authors, link)
  that a live source actually returned. If no confident match is found, the
  UI says so explicitly instead of inventing one.
- The subject-area classifier (k-NN, k=15) is trained on those derived
  labels. Because the data is skewed, it leans toward "Computer Vision" /
  "Machine Learning" for ambiguous queries. It has not been formally
  evaluated.

## Project structure

```
paper_recommender/
├── app.py                     Flask routes
├── config.py                  Environment-driven configuration
├── requirements.txt
├── .env.example
├── data/
│   ├── arxiv_papers.csv       Kaggle arXiv dataset (generated, git-ignored)
│   ├── sample_papers.csv      20-row fallback for quick tests
│   └── README_DATA.md         Dataset schema + how to swap it in
├── modules/
│   ├── preprocess.py          Lightweight text cleaning + validation
│   ├── data_loader.py         CSV loading, column mapping, dedup
│   ├── embeddings.py          MiniLM embeddings + disk caching
│   ├── similarity.py          Cosine similarity ranking
│   ├── subject_predictor.py   k-NN subject-area classifier
│   ├── ieee_retrieval.py      Crossref / IEEE verification of dataset papers
│   └── live_search.py         Live search: IEEE, OpenAlex, arXiv, Semantic Scholar
├── scripts/
│   └── prepare_kaggle_dataset.py  Downloads + converts the Kaggle dataset
├── templates/                 Jinja2 templates
├── static/css/style.css
├── static/js/main.js
└── cache/                     Embeddings + classifier cache (auto-created)
```

## Live platform search (IEEE Xplore, OpenAlex, arXiv, Semantic Scholar)

Besides the offline Kaggle dataset, every search can also query existing
research platforms live (choose them under **Search in** on the home page):

| Platform | How | Key needed? |
|---|---|---|
| IEEE Xplore | Official IEEE Xplore API if `IEEE_API_KEY` is set; otherwise OpenAlex filtered to the IEEE publisher (real IEEE papers, DOIs `10.1109/...`) | No (key optional) |
| OpenAlex | 250M+ works from all publishers (ACM, Springer, Elsevier, ...) | No |
| arXiv | arXiv API | No |
| Semantic Scholar | Graph API — off by default, since it is usually rate-limited without a key | Optional `SEMANTIC_SCHOLAR_API_KEY` |

How it stays *semantic*: each platform's search API is used only to gather
~20 candidate papers (title + abstract + metadata). Their keyword ranking
is thrown away; every candidate's title + abstract is embedded with the same
MiniLM model and scored by cosine similarity against your query, in **one
ranking together with the Kaggle papers**. The top `TOP_K` (default 10) are
shown, each labelled with where it was found. Live candidates also get a
predicted subject area from the k-NN classifier. If a platform times out or
rate-limits, it is skipped for 2 minutes so it doesn't slow every search.

Live results are fetched in parallel (typically 2-4 s total) and cached in
memory for an hour per query. If a platform fails, times out or is
rate-limited, the results page says so and the other sources still work.

## Setup

1. **Create a virtual environment and install dependencies.**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   `sentence-transformers` and `torch` are sizeable downloads; the first
   `pip install` and the first app run (which downloads the MiniLM model
   weights) both need an internet connection.

2. **Configure environment variables.**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and set `FLASK_SECRET_KEY` to a random string. Leave
   `IEEE_API_KEY` blank to use the free Crossref fallback, or add a real key
   from https://developer.ieee.org/ to use the official IEEE Xplore API.

3. **Download the Kaggle dataset.**
   ```bash
   python scripts/prepare_kaggle_dataset.py            # ~41k papers
   python scripts/prepare_kaggle_dataset.py --limit 5000   # smaller/faster
   ```
   This uses `kagglehub`; the dataset is public, so no Kaggle account is
   normally needed. It writes `data/arxiv_papers.csv`.

4. **Run the app.**
   ```bash
   python app.py
   ```
   Visit http://127.0.0.1:5000. The very first request will take longer
   while MiniLM encodes the dataset; after that, embeddings are cached in
   `cache/` and reused.

## Testing

Manual checks worth doing after setup:

- Submit each of the five example topics shown on the home page and confirm
  the top 5 results look conceptually related, not just keyword-matched.
- Submit a query shorter than 5 characters and confirm you get a validation
  message instead of a crash.
- Toggle the "Search in" sources (e.g. IEEE only, dataset only) and confirm
  every result card's source badge matches; deselect all and confirm you get
  a message instead of an error.
- Check `/health` and confirm `papers_loaded` matches your dataset size and
  `ieee_api_key_configured` reflects whether you set a key.
- Open a result's "View Details" page and confirm the DOI/author fields say
  "not available" rather than showing placeholder text, when the live
  lookup genuinely found nothing.
- Turn off your internet connection and submit a query: results should still
  render, each labeled "Unverified - no confident match found," not crash.

If you have a labeled relevance dataset (query -> known-relevant paper ids),
compute Precision@5 / Recall@5 / NDCG@5 by calling `rank_papers` directly in
a small script and comparing its output against your labels; this isn't
wired into the UI since no such labeled set was provided.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: sentence_transformers` | Dependencies not installed — re-run `pip install -r requirements.txt` inside the activated venv. |
| First request hangs for a long time | MiniLM is downloading its weights; this only happens once. Check your internet connection if it never finishes. |
| `FileNotFoundError: Dataset not found` | You haven't run `python scripts/prepare_kaggle_dataset.py` yet, or `PAPERS_CSV_PATH` in `.env` points to a missing file. |
| Every result shows "Unverified" | No internet access to `api.crossref.org` / IEEE's API, or a firewall is blocking outbound HTTPS. The app degrades gracefully by design rather than fabricating metadata. |
| Subject prediction always says "Unclassified" | Your dataset's `subject_area` column is empty or has fewer than 2 distinct labels — the classifier needs labeled data to train on. |
| Changing the dataset doesn't change results | Delete the `cache/` folder's `.npy`/`.joblib` files so embeddings and the classifier rebuild from the new data. |
| `429` or slow responses from Crossref | Crossref's public API is rate-limited for anonymous use; this is expected under heavy testing, not a bug. |

## What's intentionally left as future work

- Pagination / infinite scroll for large datasets.
- More platforms needing keys (Springer Nature, CORE, Scopus).
- An IEEE-exclusive bulk dataset (the Kaggle dataset used is arXiv).
- Automated ranking-quality evaluation (needs a labeled relevance set).
- User accounts, saved searches, or citation export.

## Deploying (free public link on Hugging Face Spaces)

1. Create a free account at https://huggingface.co/join and a **write**
   token at https://huggingface.co/settings/tokens.
2. From this folder:
   ```bash
   source venv/bin/activate
   pip install huggingface_hub
   HF_TOKEN=hf_your_token python deploy/deploy_to_hf.py <your-username>
   ```
3. Wait ~5-10 minutes for the first build. The site is then live at
   `https://<your-username>-scholarmatch.hf.space`.

The `Dockerfile` installs CPU-only PyTorch, bakes in the MiniLM model, and
serves the app with gunicorn on port 7860 with debug mode off. The dataset
CSV and the cached embeddings are uploaded with the code (Git LFS), so the
Space never has to re-encode the 41k papers.
