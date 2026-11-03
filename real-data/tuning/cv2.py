import sys, pickle, os, math, re
SP = os.environ["SCRATCH"]  # directory holding squad/dev-v2.0.json and the cached candidates (tune_cands.pkl)
sys.path.insert(0,"/Users/asp/code/freelance/proofs/rag-reliability-lab/real-data")
import run_eval_v2 as R
from rag2 import FEATURES

qs,c=pickle.load(open(SP+"/tune_cands.pkl","rb"))
arts=sorted({re.sub(r"_p\d+$","",q["own"]) for q in qs})
fold={a:i%2 for i,a in enumerate(arts)}
def prep(idx_cols):
    out=[]
    for q,cc in zip(qs,c):
        if cc is None: continue
        x=[cc[2][j] for j in idx_cols]
        out.append((q,x,R.contains(cc[0],q["golds"]),fold[re.sub(r"_p\d+$","",q["own"])]))
    return out
def probs(w,mean,std,x):
    z=w[0]+sum(w[j]*(x[j]-mean[j])/std[j] for j in range(1,len(x)))
    return 1/(1+math.exp(-z))
def run(cols,label):
    data=prep(cols)
    res=[]
    for f in (0,1):
        tr=[d for d in data if d[3]!=f]; te=[d for d in data if d[3]==f]
        nA=sum(not d[0]["imp"] for d in tr); nU=sum(d[0]["imp"] for d in tr)
        rows=[];ys=[];ws=[]
        for q,x,ok,_ in tr:
            if q["imp"]: rows.append(x);ys.append(0.);ws.append(1/nU)
            elif ok: rows.append(x);ys.append(1.);ws.append(1/nA)
        w,m,s=R.fit_logistic(rows,ys,ws)
        res+= [(probs(w,m,s,x),q["imp"],ok) for q,x,ok,_ in te]
    nA=sum(not r[1] for r in res); nU=sum(r[1] for r in res)
    # area-ish: answerable-correct at unanswerable-refused 60% and 80%; best mean
    out=[]
    best=0
    pts=[]
    for t in [i/100 for i in range(1,100)]:
        ac=sum((not r[1]) and r[0]>=t and r[2] for r in res)/nA
        ur=sum(r[1] and r[0]<t for r in res)/nU
        pts.append((ur,ac)); best=max(best,(ac+ur)/2)
    def at(u): return max(ac for ur,ac in pts if ur>=u)
    print(f"{label:28s} best mean {best:.3f}  ac@ur60 {at(.6):.3f}  ac@ur70 {at(.7):.3f} ac@ur80 {at(.8):.3f}")
n=len(FEATURES)
run(list(range(10)),"base10")
run(list(range(n)),"all18")
for j in range(10,n):
    run(list(range(10))+[j],"base10+"+FEATURES[j])
