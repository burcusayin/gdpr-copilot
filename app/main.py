# app/main.py
import os, mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import traceback

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
    try:
        model = _get_model()
        out = model.predict({"question": q.question})
        return out if isinstance(out, dict) else {"result": out}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "type": e.__class__.__name__,
                "trace": traceback.format_exc(),
            },
        )
