"""v2 real-data tools. Usage: python3 -I run_eval_v2.py <dev-v2.0.json> --split tune|test [--fit] [--out params.json]
  --split tune : articles 21 to 35 (every article after the first 20; the file has only 35). Used for all tuning.
  --split test : articles 1 to 20, the frozen test set. Run ONCE, after PROTOCOL-v2.md and the frozen weights.
Same scoring rules as run_eval.py (PROTOCOL.md)."""
import json, math, os, random, re, string, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src")); sys.path.insert(0, HERE)
from evaluate import wilson  # noqa: E402
import rag2  # noqa: E402
from rag2 import RAGv2, FEATURES  # noqa: E402

def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def contains(text, golds):
    t = " " + norm(text) + " "
    return any(g and (" " + g + " ") in t for g in golds)

def load(path, split):
    data = json.load(open(path, encoding="utf-8"))["data"]
    d = data[:20] if split == "test" else data[20:]
    docs, qs = [], []
    for a in d:
        slug = re.sub(r"\W", "_", a["title"])
        for pi, p in enumerate(a["paragraphs"]):
            did = f"{slug}_p{pi}"
            docs.append((did, " ".join(p["context"].split())))
            for q in p["qas"]:
                golds = sorted({norm(x["text"]) for x in q.get("answers", [])})
                qs.append({"id": q["id"], "q": q["question"], "imp": q["is_impossible"], "golds": golds, "own": did})
    return docs, qs

def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k/n:.1%} (95% CI {lo:.1%}-{hi:.1%})"

def candidates(system, qs):
    """Retrieval only (no gate): for each question the best sentence, its doc and the gate features."""
    out = []
    for q in qs:
        c = system.candidate(q["q"])
        out.append(c)
    return out

def fit_logistic(rows, labels, weights, l2=1.0, iters=400, lr=0.5):
    """Weighted logistic regression by full-batch gradient descent on standardised features. Returns (w, mean, std)."""
    m = len(rows[0])
    mean = [0.0] * m; std = [1.0] * m
    wsum = sum(weights)
    for j in range(1, m):
        mean[j] = sum(w * r[j] for r, w in zip(rows, weights)) / wsum
        var = sum(w * (r[j] - mean[j]) ** 2 for r, w in zip(rows, weights)) / wsum
        std[j] = math.sqrt(var) or 1.0
    Z = [[1.0] + [(r[j] - mean[j]) / std[j] for j in range(1, m)] for r in rows]
    w = [0.0] * m
    for _ in range(iters):
        g = [0.0] * m
        for z, y, sw in zip(Z, labels, weights):
            s = sum(a * b for a, b in zip(w, z))
            p = 1 / (1 + math.exp(-max(-30, min(30, s))))
            e = (p - y) * sw / wsum
            for j in range(m):
                g[j] += e * z[j]
        for j in range(m):
            w[j] -= lr * (g[j] + (l2 / wsum * 0 if j == 0 else l2 * 1e-3 * w[j]))
    return w, mean, std

def evaluate_gate(system, qs, cands, gate, thr):
    """Counts at one threshold with a given gate: (answerable_correct, nA, unanswerable_refused, nU)."""
    system.gate = {**gate, "threshold": thr}
    ac = ur = nA = nU = 0
    for q, c in zip(qs, cands):
        if c is None:
            ab, txt = True, ""
        else:
            ab = system.probability(c[2]) < thr
            txt = c[0]
        if q["imp"]:
            nU += 1; ur += ab
        else:
            nA += 1; ac += (not ab) and contains(txt, q["golds"])
    return ac, nA, ur, nU

def main():
    path = sys.argv[1]
    split = sys.argv[sys.argv.index("--split") + 1]
    docs, qs = load(path, split)
    nA = sum(not q["imp"] for q in qs)
    print(f"split {split}: documents {len(docs)}; questions {len(qs)}; answerable {nA}; unanswerable {len(qs)-nA}")
    if split == "test" or "--report" in sys.argv:   # --report on the tune split is a dry run of the report code
        import report_v2
        return report_v2.run(docs, qs)
    # ---------- tuning (tune split only) ----------
    retr_grid = [dict(k1=k1, b=b, w_par=wp, w_type=wt) for k1 in (0.9, 1.2, 1.8) for b in (0.5, 0.75)
                 for wp in (0.0, 0.3, 0.6, 1.0) for wt in (0.0, 0.1, 0.2)]
    best = None
    print("RETRIEVAL GRID (forced to answer, answerable questions only; top-1 sentence contains a gold answer)")
    A = [q for q in qs if not q["imp"]]
    for rp in retr_grid:
        sy = RAGv2(docs, retrieval=rp)
        hit = sum(1 for q in A if (lambda c: c is not None and contains(c[0], q["golds"]))(sy.candidate(q["q"])))
        print(f"  {rp} -> {hit}/{len(A)} = {hit/len(A):.1%}")
        if best is None or hit > best[0]:
            best = (hit, rp)
    print(f"CHOSEN RETRIEVAL {best[1]} ({best[0]}/{len(A)})")
    system = RAGv2(docs, retrieval=best[1])
    cands = candidates(system, qs)
    # training rows: positives = answerable and top sentence correct (weight 1/nA); negatives = unanswerable (weight 1/nU).
    # Answerable questions whose top sentence is wrong are left out: answering or refusing costs the same there.
    nU = len(qs) - len(A)
    rows, ys, ws = [], [], []
    for q, c in zip(qs, cands):
        if c is None:
            continue
        if q["imp"]:
            rows.append(c[2]); ys.append(0.0); ws.append(1.0 / nU)
        elif contains(c[0], q["golds"]):
            rows.append(c[2]); ys.append(1.0); ws.append(1.0 / len(A))
    print(f"gate training rows {len(rows)} (positives {int(sum(ys))}, negatives {len(ys)-int(sum(ys))})")
    w, mean, std = fit_logistic(rows, ys, ws)
    gate = {"mean": mean, "std": std, "w": w}
    print("FITTED WEIGHTS (standardised features):")
    for n, x in zip(FEATURES, w):
        print(f"  {n:12s} {x:+.4f}")
    # threshold: maximise the mean of the two rates on the tuning set
    print("THRESHOLD CURVE ON THE TUNING SET")
    bestt = None
    for t in [i / 100 for i in range(5, 96, 5)]:
        ac, a, ur, u = evaluate_gate(system, qs, cands, gate, t)
        bal = (ac / a + ur / u) / 2
        print(f"  thr={t:.2f}: answerable correct {ac/a:.1%} | unanswerable refused {ur/u:.1%} | mean of the two {bal:.1%}")
        if bestt is None or bal > bestt[0]:
            bestt = (bal, t)
    gate["threshold"] = bestt[1]
    print(f"CHOSEN THRESHOLD {bestt[1]} (mean of the two rates {bestt[0]:.1%} on tuning)")
    ac, a, ur, u = evaluate_gate(system, qs, cands, gate, bestt[1])
    print(f"TUNING-SET RESULT (optimistic, tuned here): answerable correct {fmt(ac, a)}; unanswerable refused {fmt(ur, u)}")
    out = {"retrieval": best[1], "gate": gate, "features": FEATURES}
    if "--out" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
        print("wrote", sys.argv[sys.argv.index("--out") + 1])

if __name__ == "__main__":
    main()
