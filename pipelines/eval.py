import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root (.. from pipelines/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import argparse, json, pandas as pd, numpy as np, pathlib, mlflow
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from models.rag_mlflow.model import RagPyFunc
from mlflow.tracking import MlflowClient


ap=argparse.ArgumentParser()
ap.add_argument("--q"), ap.add_argument("--out"), ap.add_argument("--emb", default="data/embeddings.parquet")
ap.add_argument("--topk",type=int,default=8)
a=ap.parse_args()


df=pd.read_parquet(a.emb)
X=np.stack(df["embedding"])
nn=NearestNeighbors(n_neighbors=a.topk, metric="cosine").fit(X)
queries=[json.loads(l) for l in pathlib.Path(a.q).read_text().splitlines() if l.strip()]
enc=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


rec=[]
for q in queries:
    qv=enc.encode([q["question"]],normalize_embeddings=True)
    _, idx = nn.kneighbors(qv)
    texts=df.iloc[idx[0]]["text"].tolist()
    rel=any(any(ans.lower() in t.lower() for ans in q["answers"]) for t in texts)
    rec.append(1.0 if rel else 0.0)
    
metric_key = f"recall_at_{a.topk}"
metrics = {metric_key: float(np.mean(rec))}


with mlflow.start_run():
    for k,v in metrics.items(): 
        mlflow.log_metric(k,v)
    pathlib.Path(a.out).write_text(json.dumps(metrics,indent=2))
    mlflow.log_artifact(a.out)
    
    # 1) log model as a run artifact
    model_info = mlflow.pyfunc.log_model(
        artifact_path="rag_model",
        python_model=RagPyFunc(),
        code_paths=["models", "app", "pipelines"],     # helps portable loading
        pip_requirements="requirements.txt",
    )

    # 2) create (if missing) and register a new version
    client = MlflowClient()
    model_name = "rag_pipeline"
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass  # already exists

    # NOTE: registering from the run's model_uri is more reliable than runs:/ URIs
    mv = client.create_model_version(
        name=model_name,
        source=model_info.model_uri,
        run_id=mlflow.active_run().info.run_id,
    )

    # 3) auto-promote to Production (archive old versions)
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"[MLflow] Registered {model_name} v{mv.version} and promoted to Production.")
