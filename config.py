"""
Central configuration. Reads everything from environment variables
(loaded from a .env file in development) so no credentials are hardcoded.
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

    # Kaggle arXiv dataset (built by scripts/prepare_kaggle_dataset.py).
    # Relative paths are resolved against this folder, not the shell's cwd.
    PAPERS_CSV_PATH = os.path.join(
        BASE_DIR, os.environ.get("PAPERS_CSV_PATH", os.path.join("data", "arxiv_papers.csv"))
    )
    CACHE_DIR = os.path.join(BASE_DIR, "cache")

    EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    IEEE_API_KEY = os.environ.get("IEEE_API_KEY", "").strip()
    # Optional: Semantic Scholar works without a key but is heavily rate-limited.
    SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()

    # Live platform search (IEEE / OpenAlex / arXiv / Semantic Scholar)
    LIVE_RESULTS_PER_PLATFORM = int(os.environ.get("LIVE_RESULTS_PER_PLATFORM", "20"))
    LIVE_SEARCH_TIMEOUT_SECONDS = 10

    TOP_K = int(os.environ.get("TOP_K", "10"))
    MIN_QUERY_LENGTH = 5
    REQUEST_TIMEOUT_SECONDS = 8
