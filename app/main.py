import os, mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="GDPR Copilot")
MODEL_URI = os.getenv("MODEL_URI","models:/rag_pipeline/Production")
rag = None

class QueryIn(BaseModel):
    question: str

@app.on_event("startup")
def _load():
    global rag
    rag = mlflow.pyfunc.load_model(MODEL_URI)

@app.get("/healthz")
def health():
    return {"ok": True, "model": MODEL_URI}

@app.post("/query")
def query(q: QueryIn):
    return rag.predict({"question": q.question})
