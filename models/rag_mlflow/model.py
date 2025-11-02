import os, time, mlflow.pyfunc
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

class RagPyFunc(mlflow.pyfunc.PythonModel):
    
    def load_context(self, ctx):
        self.embed = SentenceTransformer(os.getenv("EMBED_MODEL","sentence-transformers/all-MiniLM-L6-v2"))
        self.topk = int(os.getenv("RETRIEVAL_TOPK","8"))
        self.collection = os.getenv("QDRANT_COLLECTION","gdpr_collection")
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL","http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

    def predict(self, context: mlflow.pyfunc.PythonModelContext,
                model_input: dict | pd.DataFrame) -> dict | list[dict]:
        q = model_input["question"] if isinstance(model_input, dict) else model_input["question"].iloc[0]
        qv = self.embed.encode([q], normalize_embeddings=True)[0].tolist()
        hits = self.client.search(collection_name=self.collection, query_vector=qv, limit=self.topk)
        texts = [h.payload.get("text","") for h in hits] or [""]
        start = time.time()
        answer = texts[0][:512]  # simple extractive fallback
        latency_ms = int((time.time() - start) * 1000)
        return {"answer": answer, "sources": texts[:3], "latency_ms": latency_ms}
