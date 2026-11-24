"""
Download the Kaggle "arXiv Paper Abstracts" dataset and convert it into the
schema the app expects (id, title, summary, subject_area, arxiv_categories).

Dataset: https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts
(~56k arXiv rows, ~41k unique papers, mostly CS / ML / statistics).

Usage (from the paper_recommender/ folder, venv activated):
    python scripts/prepare_kaggle_dataset.py              # all unique papers
    python scripts/prepare_kaggle_dataset.py --limit 5000 # smaller, faster demo

The public dataset downloads without a Kaggle account via kagglehub. If
Kaggle ever requires auth, put your API token in ~/.kaggle/kaggle.json (or
set KAGGLE_USERNAME / KAGGLE_KEY) and re-run.
"""
import argparse
import ast
import collections
import os
import sys

import pandas as pd

KAGGLE_DATASET = "spsayakpaul/arxiv-paper-abstracts"
SOURCE_FILE = "arxiv_data_210930-054931.csv"  # newer, larger of the two files

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "data", "arxiv_papers.csv")

# arXiv category code -> readable subject area
CATEGORY_NAMES = {
    "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning",
    "stat.ML": "Machine Learning",
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Natural Language Processing",
    "cs.RO": "Robotics",
    "cs.NE": "Neural & Evolutionary Computing",
    "cs.CR": "Cybersecurity",
    "cs.SI": "Social & Information Networks",
    "cs.GR": "Computer Graphics",
    "cs.MM": "Multimedia",
    "cs.IR": "Information Retrieval",
    "cs.HC": "Human-Computer Interaction",
    "cs.MA": "Multi-Agent Systems",
    "cs.DC": "Distributed Computing",
    "cs.NI": "Computer Networks",
    "cs.DB": "Database Systems",
    "cs.SE": "Software Engineering",
    "cs.CY": "Computers & Society",
    "cs.IT": "Information Theory",
    "math.IT": "Information Theory",
    "cs.SY": "Systems & Control",
    "eess.SY": "Systems & Control",
    "eess.IV": "Image & Video Processing",
    "eess.SP": "Signal Processing",
    "eess.AS": "Audio & Speech Processing",
    "cs.SD": "Audio & Speech Processing",
    "math.OC": "Optimization & Control",
    "stat.AP": "Statistics",
    "stat.ME": "Statistics",
    "stat.CO": "Statistics",
    "stat.TH": "Statistics",
    "math.ST": "Statistics",
    "q-bio.QM": "Quantitative Biology",
    "q-bio.NC": "Quantitative Biology",
}

ARCHIVE_FALLBACK = {
    "cs": "Computer Science (Other)",
    "stat": "Statistics",
    "math": "Mathematics",
    "eess": "Electrical Engineering",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "physics": "Physics",
    "astro-ph": "Physics",
    "cond-mat": "Physics",
    "quant-ph": "Physics",
    "econ": "Economics",
}

MIN_PAPERS_PER_SUBJECT = 50  # rarer subjects are folded into "Other"


def parse_terms(raw) -> list:
    try:
        terms = ast.literal_eval(raw) if isinstance(raw, str) else []
        return [t for t in terms if isinstance(t, str)]
    except (ValueError, SyntaxError):
        return []


# This dataset was scraped by querying cs.CV / cs.LG / stat.ML, so the first
# listed term is just the scrape query, not the paper's real focus. Instead we
# pick the most *specific* category: a niche one (cs.CR, cs.CL, cs.RO, ...)
# beats Computer Vision, which beats AI, which beats generic Machine Learning.
BROAD_TIERS = [
    {"cs.CV", "eess.IV"},          # tier 1: vision / imaging
    {"cs.AI"},                     # tier 2
    {"cs.LG", "stat.ML"},          # tier 3: generic ML (least specific)
]
BROAD = set().union(*BROAD_TIERS)
BROAD_TIER_SUBJECT = ["Computer Vision", "Artificial Intelligence", "Machine Learning"]


def readable(term: str) -> str:
    if term in CATEGORY_NAMES:
        return CATEGORY_NAMES[term]
    return ARCHIVE_FALLBACK.get(term.split(".")[0], "Other")


def subject_for(terms: list, term_freq: dict) -> str:
    if not terms:
        return ""
    specific = [t for t in terms if t not in BROAD]
    if specific:
        # rarest category in the corpus = most specific signal
        return readable(min(specific, key=lambda t: term_freq.get(t, 0)))
    for tier, subject in zip(BROAD_TIERS, BROAD_TIER_SUBJECT):
        if tier & set(terms):
            return subject
    return "Other"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="keep only N papers (random sample); 0 = all")
    args = parser.parse_args()

    import kagglehub  # imported here so the app itself doesn't depend on it

    print(f"Downloading Kaggle dataset '{KAGGLE_DATASET}' ...")
    dataset_dir = kagglehub.dataset_download(KAGGLE_DATASET)
    source_path = os.path.join(dataset_dir, SOURCE_FILE)
    if not os.path.exists(source_path):
        sys.exit(f"Expected {SOURCE_FILE} in {dataset_dir}, found: {os.listdir(dataset_dir)}")

    df = pd.read_csv(source_path)
    print(f"Loaded {len(df):,} raw rows from {SOURCE_FILE}")

    df = df.rename(columns={"titles": "title", "abstracts": "summary"})
    df["title"] = df["title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["summary"] = df["summary"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[(df["title"].str.len() > 0) & (df["summary"].str.len() > 20)]

    # The source has many duplicate rows (same paper scraped under several queries)
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    term_lists = df["terms"].apply(parse_terms)
    df["arxiv_categories"] = term_lists.apply(lambda ts: " ".join(ts))
    term_freq = collections.Counter(t for ts in term_lists for t in ts)
    df["subject_area"] = term_lists.apply(lambda ts: subject_for(ts, term_freq))

    counts = df["subject_area"].value_counts()
    rare = counts[counts < MIN_PAPERS_PER_SUBJECT].index
    df.loc[df["subject_area"].isin(rare), "subject_area"] = "Other"

    if args.limit and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=42).reset_index(drop=True)

    df.insert(0, "id", range(1, len(df) + 1))
    df = df[["id", "title", "summary", "subject_area", "arxiv_categories"]]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Wrote {len(df):,} unique papers to {args.output}")
    print("Subject areas:")
    print(df["subject_area"].value_counts().to_string())


if __name__ == "__main__":
    main()
