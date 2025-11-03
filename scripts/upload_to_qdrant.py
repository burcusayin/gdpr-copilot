#!/usr/bin/env python3
import os, pandas as pd, numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

URL = os.environ["QDRANT_URL"]
KEY = os.environ["QDRANT_API_KEY"]
COL = os.getenv("QDRANT_COLLECTION", "gdpr_collection")

df = pd.read_parquet("data/embeddings.parquet")
vec_col = "embedding" if "embedding" in df.columns else ("vector" if "vector" in df.columns else None)
assert vec_col, f"No embedding/vector column found in {df.columns}"

vectors = df[vec_col].tolist()
if isinstance(vectors[0], np.ndarray):
    vectors = [v.astype(float).tolist() for v in vectors]
dim = len(vectors[0])

cli = QdrantClient(url=URL, api_key=KEY)
try:
    cli.get_collection(COL)
except Exception:
    cli.create_collection(COL, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

ids = df["id"].tolist() if "id" in df.columns else list(range(1, len(vectors)+1))
payloads = [{"text": str(t)} for t in df["text"]] if "text" in df.columns else [{} for _ in vectors]
points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(vectors))]

BATCH=100
for i in range(0, len(points), BATCH):
    cli.upsert(collection_name=COL, points=points[i:i+BATCH])

print("Count:", cli.count(COL, exact=True).count)
