import sys, pickle, os, math, re, time
SP = os.environ["SCRATCH"]  # directory holding squad/dev-v2.0.json and the cached candidates (tune_cands.pkl)
sys.path.insert(0,"/Users/asp/code/freelance/proofs/rag-reliability-lab/real-data")
import run_eval_v2 as R
from gam import fit_stumps, predict

qs,c=pickle.load(open(SP+"/tune_cands.pkl","rb"))
arts=sorted({re.sub(r"_p\d+$","",q["own"]) for q in qs})
fold={a:i%2 for i,a in enumerate(arts)}
data=[(q,cc[2][1:],R.contains(cc[0],q["golds"]),fold[re.sub(r"_p\d+$","",q["own"])]) for q,cc in zip(qs,c) if cc]
for rounds,lr in ((100,0.15),(250,0.1)):
    res=[]
    t0=time.time()
    for f in (0,1):
        tr=[d for d in data if d[3]!=f]; te=[d for d in data if d[3]==f]
        nA=sum(not d[0]["imp"] for d in tr); nU=sum(d[0]["imp"] for d in tr)
        X=[];y=[];w=[]
        for q,x,ok,_ in tr:
            if q["imp"]: X.append(x);y.append(0.);w.append(1/nU)
            elif ok: X.append(x);y.append(1.);w.append(1/nA)
        sw=sum(w); w=[wi/sw*len(w) for wi in w]
        m=fit_stumps(X,y,w,rounds=rounds,lr=lr)
        res+=[(predict(m,x),q["imp"],ok) for q,x,ok,_ in te]
    nA=sum(not r[1] for r in res); nU=sum(r[1] for r in res)
    pts=[];best=0
    for t in [i/100 for i in range(1,100)]:
        ac=sum((not r[1]) and r[0]>=t and r[2] for r in res)/nA
        ur=sum(r[1] and r[0]<t for r in res)/nU
        pts.append((ur,ac)); best=max(best,(ac+ur)/2)
    at=lambda u:max(ac for ur,ac in pts if ur>=u)
    print(f"stumps r={rounds} lr={lr}: best mean {best:.3f} ac@ur60 {at(.6):.3f} ac@ur70 {at(.7):.3f} ac@ur80 {at(.8):.3f} ({time.time()-t0:.0f}s)")
