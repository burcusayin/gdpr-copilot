import argparse, pathlib, re, shutil


def clean(t): 
    return re.sub(r"\s+"," ",t).strip()

def chunk(t, size=400, overlap=120):
    out=[]; i=0
    while i<len(t): 
        out.append(t[i:i+size]) 
        i += max(1,size-overlap)
    return out

ap=argparse.ArgumentParser()
ap.add_argument("--src"), ap.add_argument("--out")
ap.add_argument("--chunk_size",type=int,default=400)
ap.add_argument("--chunk_overlap",type=int,default=120)
a=ap.parse_args()
out=pathlib.Path(a.out); 

if out.exists(): 
    shutil.rmtree(out)
    
out.mkdir(parents=True,exist_ok=True)

doc=0
for p in pathlib.Path(a.src).glob("*.md"):
    text=clean(p.read_text(encoding="utf-8"))
    for i,c in enumerate(chunk(text,a.chunk_size,a.chunk_overlap)):
        (out/f"doc{doc}_chunk{i}.txt").write_text(c,encoding="utf-8")
    doc+=1
