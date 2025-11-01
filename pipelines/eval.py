import numpy as np
import argparse, json, pandas as pd, numpy as np, pathlib, mlflow
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


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
    
metrics={"recall@{}".format(a.topk): float(np.mean(rec))}

with mlflow.start_run():
    for k,v in metrics.items(): 
        mlflow.log_metric(k,v)
    pathlib.Path(a.out).write_text(json.dumps(metrics,indent=2))
    mlflow.log_artifact(a.out)
