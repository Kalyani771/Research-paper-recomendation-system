# Production image for Hugging Face Spaces (Docker SDK) or any Docker host.
# Serves the Flask app with gunicorn on port 7860 (Hugging Face's default).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    FLASK_DEBUG=0 \
    PORT=7860

WORKDIR /app

# CPU-only PyTorch first (the default wheel bundles ~2.5 GB of unused CUDA libraries)
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==22.0.0

# Bake the MiniLM model into the image so the first search doesn't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# App code + the Kaggle dataset + precomputed embeddings (encoding 41k papers
# from scratch takes ~30 min on CPU, so the cached .npy file must be included)
COPY . .

# Hugging Face runs containers as user 1000; the app writes its classifier cache
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860
CMD gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120
