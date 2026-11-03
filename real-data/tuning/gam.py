import math
def fit_stumps(X, y, w, rounds=150, lr=0.15, lam=1.0, min_leaf=1.0):
    """Boosted decision stumps (additive model), weighted logistic loss. X rows without bias. Returns (base, stumps[(j,thr,left,right)])."""
    n=len(X); m=len(X[0]); W=sum(w)
    pw=sum(wi*yi for wi,yi in zip(w,y))/W
    base=math.log(pw/(1-pw))
    order=[sorted(range(n),key=lambda i:X[i][j]) for j in range(m)]
    F=[base]*n; stumps=[]
    for _ in range(rounds):
        g=[];h=[]
        for i in range(n):
            p=1/(1+math.exp(-F[i])); g.append((p-y[i])*w[i]*n/W if False else (p-y[i])*w[i]); h.append(max(p*(1-p)*w[i],1e-12))
        G=sum(g);H=sum(h)
        best=None
        for j in range(m):
            gl=hl=0.0
            o=order[j]
            for k in range(n-1):
                i=o[k]; gl+=g[i]; hl+=h[i]
                if X[o[k]][j]==X[o[k+1]][j]: continue
                gr=G-gl; hr=H-hl
                if hl<min_leaf*1e-4*H or hr<min_leaf*1e-4*H: continue
                gain=gl*gl/(hl+lam*1e-4*H)+gr*gr/(hr+lam*1e-4*H)
                if best is None or gain>best[0]:
                    best=(gain,j,(X[o[k]][j]+X[o[k+1]][j])/2,-gl/(hl+lam*1e-4*H),-gr/(hr+lam*1e-4*H))
        _,j,t,lv,rv=best
        lv*=lr; rv*=lr
        stumps.append((j,t,lv,rv))
        for i in range(n):
            F[i]+= lv if X[i][j]<=t else rv
    return base,stumps
def predict(model,x):
    base,stumps=model
    z=base+sum((lv if x[j]<=t else rv) for j,t,lv,rv in stumps)
    return 1/(1+math.exp(-z))
