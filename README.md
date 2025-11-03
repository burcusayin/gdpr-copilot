# GDPR Copilot — RAG + MLflow + Qdrant + Docker

A preliminary app for real-world, production-minded RAG service:

- **Pipeline** with DVC: `ingest → embed → eval (MLflow) → index (Qdrant)`
- **Model registry**: MLflow `pyfunc` model, loaded by a FastAPI microservice
- **Vector DB**: Qdrant (local via Docker, or Qdrant Cloud Hobby)
- **Containers**: Docker + Compose; easy path to **Cloud Run** deploy
- **Traceability**: metrics + artifacts in MLflow; versioned code & data

---

## Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Project layout](#project-layout)
- [Quickstart (local)](#quickstart-local)
- [Detailed steps](#detailed-steps)
- [Common commands](#common-commands)
- [Environment variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Deploy (free tier path)](#deploy-free-tier-path)
- [License](#license)

---

## Architecture

```
[DVC] ingest → embed → eval (MLflow logs & registers) → index (Qdrant)
                                   │
                          models:/rag_pipeline_py311@prod
                                   │
                     FastAPI (mlflow.pyfunc.load_model)
                                   │
                        Qdrant vector search (top-k)
```

---

## Prerequisites

- **OS**: macOS / Linux
- **Python**: 3.11 (recommended; aligns with the app container)
- **Docker**: Docker Desktop (Compose v2 → `docker compose …`)
- **Pip** + **virtualenv**
- **DVC** (installed via `pip`)

Optional (for cloud):
- **Qdrant Cloud** Hobby cluster
- **DagsHub** repo with MLflow enabled
- **gcloud** CLI for Cloud Run

---

## Project layout

```
.
├── app/
│   └── main.py                 # FastAPI service loading MLflow pyfunc
├── models/
│   └── rag_mlflow/model.py     # RagPyFunc (SentenceTransformers + Qdrant)
├── pipelines/
│   ├── ingest.py               # load/clean GDPR docs
│   ├── embed.py                # create embeddings parquet
│   ├── eval.py                 # logs metrics + registers model in MLflow
│   └── index.py                # push vectors to Qdrant
├── scripts/
│   ├── register_and_promote.py # one-off MLflow registrar (sets alias)
│   └── smoke.sh                # local health + sample query
├── data/
│   ├── raw_docs/               # source GDPR snippets/articles
│   ├── clean_docs/             # cleaned text
│   └── embeddings.parquet
├── dvc.yaml
├── docker-compose.yml
├── requirements.txt
├── Makefile
└── README.md
```

---

## Quickstart (local)

```bash
# 1) Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Bring up Qdrant (6333) + MLflow (host port 5000)
docker compose up -d qdrant mlflow

# 3) Run the pipeline and register a model
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
dvc repro              # ingest → embed → eval → index
# If eval stage was previously cached, force a registration:
python scripts/register_and_promote.py

# 4) Start the API
docker compose up -d app

# 5) Smoke test
curl -s http://localhost:8080/healthz
curl -s -X POST http://localhost:8080/query -H 'Content-Type: application/json' \
  -d '{"question":"What are the data processing principles?"}'
```

---

## Detailed steps

### 1) Set up Python & deps
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
> This repo assumes **Python 3.11** for best compatibility with the app container.

### 2) Start infra locally
```bash
docker compose up -d qdrant mlflow
# MLflow UI: http://localhost:5000
```
Compose config uses:
- **Persistent registry DB** at `./mlflow/mlflow.db`
- **Proxied artifacts** (`--serve-artifacts`) stored in `./artifacts`
- **Allowed hosts** include `mlflow:5000`, `localhost`, `127.0.0.1`

### 3) Run the DVC pipeline
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
dvc repro
```
- `ingest`: loads/cleans GDPR docs into `data/clean_docs`
- `embed`: writes `data/embeddings.parquet` (requires `pyarrow`)
- `eval`: logs metrics to MLflow; **also registers** a `pyfunc` model
- `index`: pushes vectors + payloads to Qdrant

> If `eval` is skipped, run the registrar: `python scripts/register_and_promote.py`

### 4) Model registry: name + alias
The service expects an alias so you can update models without changing envs:

- **Model name**: `rag_pipeline_py311`
- **Alias**: `prod`

Use the included script to (re)register & alias:
```bash
python scripts/register_and_promote.py
# or from inside the app container:
docker compose exec -T app python scripts/register_and_promote.py
```

### 5) Start the API
```bash
docker compose up -d app
curl -s http://localhost:8080/healthz
curl -s -X POST http://localhost:8080/query -H 'Content-Type: application/json' \
  -d '{"question":"What are the data processing principles?"}'
```
The API lazy-loads the MLflow model on the **first** `/query`, so the very first call may take longer (downloads model weights, etc.).

---

## Common commands

```bash
# run the whole pipeline (host)
dvc repro

# index only (host)
dvc repro dvc.yaml:index

# full stack up / down
docker compose up -d qdrant mlflow app
docker compose down

# tail app logs
docker compose logs -f app

# quick smoke test
make smoke
# or
./scripts/smoke.sh
```

---

## Environment variables

**Compose (local):**
- `MLFLOW_TRACKING_URI=http://mlflow:5000` (inside the network)
- `MODEL_URI=models:/rag_pipeline_py311@prod`
- `QDRANT_URL=http://qdrant:6333`
- `TOKENIZERS_PARALLELISM=false`

**Host (CLI tools / DVC):**
- `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`
- `QDRANT_URL=http://127.0.0.1:6333` (only if running host-side tools against Qdrant)
- `QDRANT_API_KEY` if using Qdrant Cloud

**Cloud Run (example):**
- `MODEL_URI=models:/rag_pipeline_py311@prod`
- `MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow`
- `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` (if your MLflow is auth-protected)
- `QDRANT_URL=https://<your-qdrant-cloud-url>`
- `QDRANT_API_KEY=<your-key>`
- `TOKENIZERS_PARALLELISM=false`

---

## Troubleshooting

- **`docker-compose` not found** → Use **`docker compose`** (space, v2).
- **Docker daemon not running** → Start Docker Desktop.
- **Port 5000 already in use** → Change MLflow host port to `5001:5000` in `docker-compose.yml`.
- **MLflow “Invalid Host header”** → Compose uses hostname `mlflow`; compose config already sets `--allowed-hosts "mlflow:5000,localhost:*,127.0.0.1:*"`.
- **`/artifacts` read-only** → We use `--serve-artifacts --artifacts-destination /artifacts` to let the server handle uploads.
- **Model registry empty / “not found”** → Run `python scripts/register_and_promote.py` (host or in container). Prefer **aliases** (e.g., `@prod`) over deprecated stages.
- **Container missing deps** (e.g., `ModuleNotFoundError: qdrant_client`) → Ensure `requirements.txt` includes `qdrant-client`, `sentence-transformers`, `mlflow`, etc., then `docker compose build --no-cache app`.
- **Python version mismatch warning** → Register the model **inside** the container (py311) using the script above.
- **`load_context() got an unexpected keyword 'context'`** → Define `load_context(self, context=..., **_)` in `models/rag_mlflow/model.py` and re-register the model artifact.
- **First `/query` slow** → Lazy load; subsequent queries are fast.
- **TTY error** when exec’ing into containers → Use `docker compose exec -T …`.

---

## Deploy (free tier path)

### Qdrant Cloud (Hobby)
```bash
export QDRANT_URL="https://<your-qdrant-cloud-url>"
export QDRANT_API_KEY="<your-key>"
dvc repro dvc.yaml:index   # push vectors to cloud
```

### MLflow remote via DagsHub
1) Create a DagsHub repo; enable MLflow.
2) Point tools at DagsHub MLflow:
   ```bash
   export MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repo>.mlflow"
   export MLFLOW_TRACKING_USERNAME="<user>"
   export MLFLOW_TRACKING_PASSWORD="<token>"
   ```
3) Register + alias (host or inside container):
   ```bash
   python scripts/register_and_promote.py
   ```
   Use `MODEL_URI=models:/rag_pipeline_py311@prod` in production.

### Cloud Run (containerized API)
```bash
# build & push
gcloud builds submit --tag "gcr.io/$(gcloud config get-value project)/gdpr-copilot:latest"

# deploy
gcloud run deploy gdpr-copilot \
  --image "gcr.io/$(gcloud config get-value project)/gdpr-copilot:latest" \
  --region europe-west1 --platform managed --allow-unauthenticated \
  --port 8080 --memory 1Gi \
  --set-env-vars MODEL_URI="models:/rag_pipeline_py311@prod" \
  --set-env-vars MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repo>.mlflow" \
  --set-env-vars MLFLOW_TRACKING_USERNAME="<user>" \
  --set-env-vars MLFLOW_TRACKING_PASSWORD="<token>" \
  --set-env-vars QDRANT_URL="https://<your-qdrant-cloud-url>" \
  --set-env-vars QDRANT_API_KEY="<your-key>" \
  --set-env-vars TOKENIZERS_PARALLELISM=false
```

Smoke test with your Cloud Run URL:
```bash
curl -s "$CLOUD_URL/healthz"
curl -s -X POST "$CLOUD_URL/query" -H 'Content-Type: application/json' \
  -d '{"question":"What are the data processing principles?"}'
```

---

## License

MIT
