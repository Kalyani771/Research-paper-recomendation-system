"""
Loads the paper dataset from CSV, normalizes column names, drops duplicates
and rows with no usable text, and flags rows whose venue text mentions IEEE
(a hint only -- real verification happens live in ieee_retrieval.py).
"""
import logging
import os

import pandas as pd

from modules.preprocess import clean_text

logger = logging.getLogger(__name__)

# If your CSV uses different column names, map them here: csv_name -> our_name
COLUMN_ALIASES = {
    "abstract": "summary",
    "abstracts": "summary",
    "summaries": "summary",
    "description": "summary",
    "paper_title": "title",
    "titles": "title",
    "year": "year_hint",
    "venue": "venue_hint",
    "journal": "venue_hint",
    "conference": "venue_hint",
    "category": "subject_area",
    "domain": "subject_area",
}

REQUIRED_COLUMNS = ["id", "title", "summary"]
OPTIONAL_COLUMNS = ["authors_hint", "year_hint", "venue_hint", "subject_area", "doi", "arxiv_categories"]


def load_papers(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. See data/README_DATA.md for "
            "setup instructions, or check PAPERS_CSV_PATH in your .env file."
        )

    df = pd.read_csv(csv_path, dtype={"year_hint": str, "year": str, "doi": str})
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset at '{csv_path}' is missing required column(s): {missing}. "
            "See data/README_DATA.md for the expected schema."
        )

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Clean text fields
    df["title"] = df["title"].apply(clean_text)
    df["summary"] = df["summary"].apply(clean_text)
    df["authors_hint"] = df["authors_hint"].fillna("").apply(clean_text)
    df["venue_hint"] = df["venue_hint"].fillna("").apply(clean_text)
    df["subject_area"] = df["subject_area"].fillna("").apply(clean_text)
    df["arxiv_categories"] = df["arxiv_categories"].fillna("").apply(clean_text)
    # Read as text so "2019" never becomes "2019.0" when some years are missing
    df["year_hint"] = df["year_hint"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df["doi"] = df["doi"].fillna("").astype(str).str.strip()

    # Drop rows with no usable text
    before = len(df)
    df = df[(df["title"].str.len() > 0) & (df["summary"].str.len() > 0)]
    dropped_empty = before - len(df)

    # Deduplicate: prefer DOI when present, otherwise fall back to title text
    before = len(df)
    if "doi" in df.columns and df["doi"].astype(str).str.strip().ne("").any():
        has_doi = df["doi"].astype(str).str.strip().ne("")
        df_doi = df[has_doi].drop_duplicates(subset=["doi"])
        df_no_doi = df[~has_doi].drop_duplicates(subset=["title"])
        df = pd.concat([df_doi, df_no_doi], ignore_index=True)
    else:
        df = df.drop_duplicates(subset=["title"])
    dropped_dupes = before - len(df)

    df = df.reset_index(drop=True)
    df["row_index"] = df.index  # stable position used to align with embeddings

    # Hint only -- NOT a verified IEEE flag. Real verification happens live.
    df["is_ieee_hint"] = df["venue_hint"].str.contains("ieee", case=False, na=False)

    # Text used to build embeddings: title + summary carries the most meaning
    df["embedding_text"] = (df["title"] + ". " + df["summary"]).apply(clean_text)

    logger.info(
        "Loaded %d papers from %s (dropped %d empty, %d duplicate)",
        len(df), csv_path, dropped_empty, dropped_dupes,
    )
    return df
