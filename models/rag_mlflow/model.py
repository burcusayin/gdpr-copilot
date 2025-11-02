import os, time, mlflow.pyfunc
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class RagPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, ctx):
        self.embed = None
        self.embed_name = os.getenv("EMBED_MODEL","sentence-transformers/all-MiniLM-L6-v2")
        self.topk = int(os.getenv("RETRIEVAL_TOPK","8"))
        self.collection = os.getenv("QDRANT_COLLECTION","gdpr_collection")
        self.client = QdrantClient(url=os.getenv("QDRANT_URL","http://localhost:6333"),
                                   api_key=os.getenv("QDRANT_API_KEY"))
        self._collection_ready = False

    def _get_embedder(self):
        if self.embed is None:
            from sentence_transformers import SentenceTransformer
            self.embed = SentenceTransformer(self.embed_name)
        return self.embed

    def _ensure_collection(self, dim: int):
        if self._collection_ready:
            return
        try:
            self.client.get_collection(self.collection)
        except Exception:
            # create with detected dimension
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._collection_ready = True

    def predict(self, ctx, model_input):
        # accept dict or DataFrame
        q = model_input["question"] if isinstance(model_input, dict) else model_input["question"].iloc[0]
        emb = self._get_embedder()
        qv = emb.encode([q], normalize_embeddings=True)[0].tolist()

        # ensure the collection exists with the right dim before searching
        self._ensure_collection(dim=len(qv))

        hits = self.client.search(collection_name=self.collection, query_vector=qv, limit=self.topk)
        texts = [h.payload.get("text","") for h in hits] or [""]
        start = time.time()
        answer = texts[0][:512]
        latency_ms = int((time.time() - start) * 1000)
        return {"answer": answer, "sources": texts[:3], "latency_ms": latency_ms}
