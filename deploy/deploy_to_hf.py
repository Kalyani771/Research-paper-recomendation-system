"""
Deploy this app to a free Hugging Face Space (Docker SDK).

Usage (from the paper_recommender/ folder):
    pip install huggingface_hub
    HF_TOKEN=hf_xxx python deploy/deploy_to_hf.py <your-hf-username> [space-name]

The token needs "write" permission (https://huggingface.co/settings/tokens).
The site will be live at  https://<username>-<space-name>.hf.space
after the first build finishes (about 5-10 minutes).
"""
import os
import sys

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

if len(sys.argv) < 2 or not os.environ.get("HF_TOKEN"):
    sys.exit(__doc__)

user = sys.argv[1]
space = sys.argv[2] if len(sys.argv) > 2 else "scholarmatch"
repo_id = f"{user}/{space}"

for required in ("data/arxiv_papers.csv",):
    if not os.path.exists(os.path.join(ROOT, required)):
        sys.exit(f"Missing {required}: run  python scripts/prepare_kaggle_dataset.py  first.")
if not any(f.startswith("embeddings_") and f.endswith(".npy") for f in os.listdir(os.path.join(ROOT, "cache"))):
    sys.exit("Missing cached embeddings in cache/: start the app locally once so they are built.")

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

# A random secret key for Flask sessions, stored as a Space secret (not in the code)
api.add_space_secret(repo_id, "FLASK_SECRET_KEY", os.urandom(24).hex())

print(f"Uploading project to {repo_id} (large files go through Git LFS automatically)...")
api.upload_folder(
    repo_id=repo_id, repo_type="space", folder_path=ROOT,
    ignore_patterns=["venv/*", "**/__pycache__/*", "*.pyc", ".env", "cache/*.joblib",
                     "README.md", "deploy/*", "data/sample_papers.csv"],
    commit_message="Deploy ScholarMatch",
)
# The Space needs its own README with the Docker configuration header
api.upload_file(repo_id=repo_id, repo_type="space", path_or_fileobj=os.path.join(HERE, "hf_space_README.md"),
                path_in_repo="README.md", commit_message="Space configuration")

print(f"\nDone. Build progress: https://huggingface.co/spaces/{repo_id}")
print(f"Your site (after the build finishes): https://{user.lower()}-{space.lower()}.hf.space")
