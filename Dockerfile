# Example base (adjust if you already have one)
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps you *actually* need (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Force binary wheels for heavy libs (no source builds)
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --only-binary=:all: \
      numpy==1.26.4 scipy==1.11.4 scikit-learn==1.4.2 \
 && pip install --no-cache-dir -r requirements.txt \
 && rm -rf /root/.cache
