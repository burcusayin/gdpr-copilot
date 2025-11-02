# app/main.py
import os, mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="GDPR Copilot")

MODEL_URI = os.getenv("MODEL_URI", "models:/rag_pipeline/Production")
rag = None  # will be loaded on first use

class QueryIn(BaseModel):
    question: str

def _get_model():
    global rag
    if rag is None:
        rag = mlflow.pyfunc.load_model(MODEL_URI)
    return rag

@app.get("/healthz")
def health():
    # Don't force model load here; keep it lightweight
    return {"ok": True, "model": MODEL_URI if rag is not None else None}

@app.post("/query")
def query(q: QueryIn):
    model = _get_model()
    return model.predict({"question": q.question})
