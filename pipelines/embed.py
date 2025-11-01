import argparse, pathlib, pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ap=argparse.ArgumentParser()
ap.add_argument("--inp"), ap.add_argument("--out")
ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
a=ap.parse_args()

m=SentenceTransformer(a.model)

rows=[]
for p in tqdm(sorted(pathlib.Path(a.inp).glob("*.txt"))):
    t=p.read_text(encoding="utf-8")
    e=m.encode([t], normalize_embeddings=True)[0].tolist()
    rows.append({"text":t,"embedding":e})
    
df=pd.DataFrame(rows)
pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True)
df.to_parquet(a.out, index=False)
