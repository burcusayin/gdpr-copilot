import argparse, pandas as pd, os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True)
ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION","gdpr_collection"))
a = ap.parse_args()

client = QdrantClient(url=os.getenv("QDRANT_URL","http://localhost:6333"),
                      api_key=os.getenv("QDRANT_API_KEY"))

df = pd.read_parquet(a.emb)
dim = len(df["embedding"].iloc[0])

# create collection if missing
existing = [c.name for c in client.get_collections().collections]
if a.collection not in existing:
    client.recreate_collection(
        collection_name=a.collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

points = [PointStruct(id=i, vector=row["embedding"], payload={"text": row["text"]})
          for i, row in df.iterrows()]

client.upsert(collection_name=a.collection, points=points)
print(f"Indexed {len(points)} chunks into '{a.collection}'")
