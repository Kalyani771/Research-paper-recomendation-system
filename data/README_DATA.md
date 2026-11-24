# Dataset notes

## Primary dataset: Kaggle "arXiv Paper Abstracts"

Source: https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts
(file `arxiv_data_210930-054931.csv`, ~56k rows).

Build it with:

```bash
python scripts/prepare_kaggle_dataset.py              # all ~41k unique papers
python scripts/prepare_kaggle_dataset.py --limit 5000 # random 5k sample
```

The script downloads the dataset with `kagglehub` (public dataset, normally
no Kaggle login needed; if Kaggle asks for auth, put your API token in
`~/.kaggle/kaggle.json`), then writes `data/arxiv_papers.csv` with:

| column             | from                     | notes                                   |
|--------------------|--------------------------|-----------------------------------------|
| `id`               | generated (1..N)         | used in `/paper/<id>` URLs              |
| `title`            | `titles`                 | whitespace-normalized                   |
| `summary`          | `abstracts`              | the paper's real arXiv abstract         |
| `subject_area`     | derived from `terms`     | readable name, see below                |
| `arxiv_categories` | `terms`                  | raw tags, e.g. `cs.CV cs.LG`            |

Processing: duplicate titles removed (the raw file repeats many papers),
rows with empty titles/abstracts dropped.

### How `subject_area` is derived

The dataset was scraped by querying cs.CV, cs.LG and stat.ML, so the first
tag on each paper is just that query and is not meaningful. Instead, the
most specific tag wins: a niche category (e.g. `cs.CR` → Cybersecurity,
`cs.CL` → Natural Language Processing, `cs.RO` → Robotics) beats
Computer Vision, which beats Artificial Intelligence, which beats generic
Machine Learning. Subjects with fewer than 50 papers become "Other".

### Limitations

- **Not IEEE-specific** — these are arXiv preprints. IEEE status is only
  claimed when the live verification step (`modules/ieee_retrieval.py`)
  finds the published version with an IEEE publisher.
- **Skewed coverage** — ~45% Computer Vision, ~20% Machine Learning; areas
  like databases, networking or software engineering have under 100 papers.
- **No year / venue / author / DOI columns** — those fields come only from
  live verification.

## Using a different dataset

Any CSV with `id`, `title`, `summary` columns works (aliases such as
`abstract`, `abstracts`, `titles`, `category` are mapped automatically in
`modules/data_loader.py`). Point `PAPERS_CSV_PATH` in `.env` at it and
restart. Embeddings are cached by a hash of the text, so a changed dataset
re-encodes automatically on first start.

`sample_papers.csv` is the original 20-row hand-written demo set, kept for
quick tests only (its summaries are paraphrases, not real abstracts).
