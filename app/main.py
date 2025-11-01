from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="GDPR Copilot")

class QueryIn(BaseModel): question: str

@app.get("/healthz") def health(): return {"ok": True}

@app.post("/query") def query(q: QueryIn): return {"answer": f"(stub) {q.question}", "sources": [], "latency_ms": 0}
