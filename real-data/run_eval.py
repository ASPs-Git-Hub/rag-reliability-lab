"""Real-data run. Usage: python3 -I run_eval.py <dev-v2.0.json> [--sweep]. Reads the data file as JSON only."""
import json, os, random, re, string, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from rag import BrokenRAG, RAG  # noqa: E402
from evaluate import wilson  # noqa: E402

N_ART = 20

def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def contains(text, golds):
    t = " " + norm(text) + " "
    return any(g and (" " + g + " ") in t for g in golds)

def load(path):
    d = json.load(open(path, encoding="utf-8"))["data"][:N_ART]
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

def score_all(system, qs, docs_by_id, cites):
    res = []
    for q in qs:
        r = system.answer(q["q"])
        ab = r["abstained"]
        cit = r["citations"][0] if r["citations"] else None
        if q["imp"]:
            head = ab; loose = ab
        else:
            head = (not ab) and contains(r["answer"], q["golds"])
            loose = (not ab) and ((cit and contains(docs_by_id[cit], q["golds"])) if cites else head)
        res.append({"id": q["id"], "ab": ab, "cit": cit, "ans": r["answer"], "score": r["score"], "cov": r.get("coverage", 0), "head": bool(head), "loose": bool(loose)})
    return res

def summarize(name, qs, res):
    A = [r for q, r in zip(qs, res) if not q["imp"]]; U = [r for q, r in zip(qs, res) if q["imp"]]
    print(f"{name}: answerable correct (headline) {fmt(sum(r['head'] for r in A), len(A))}")
    print(f"{name}: answerable correct (looser, cited paragraph) {fmt(sum(r['loose'] for r in A), len(A))}")
    print(f"{name}: answerable refused {fmt(sum(r['ab'] for r in A), len(A))}")
    print(f"{name}: unanswerable correctly refused {fmt(sum(r['ab'] for r in U), len(U))}")
    return A, U

def main():
    path = sys.argv[1]
    docs, qs = load(path)
    dbi = dict(docs)
    nA = sum(not q["imp"] for q in qs)
    print(f"documents {len(docs)}; questions {len(qs)}; answerable {nA}; unanswerable {len(qs)-nA}")
    with open(os.path.join(HERE, "frozen_question_ids.txt"), "w") as f:
        for q in qs: f.write(("U " if q["imp"] else "A ") + q["id"] + "\n")
    if "--sweep" in sys.argv:
        print("AFTER-SEEING-RESULTS SWEEP, tuned on the test questions, not a result")
        print("min_coverage sweep at threshold 0.30")
        for c in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0):
            res = score_all(RAG(docs, min_coverage=c), qs, dbi, True)
            A = [r for q, r in zip(qs, res) if not q["imp"]]; U = [r for q, r in zip(qs, res) if q["imp"]]
            print(f"  cov={c}: answerable correct {fmt(sum(r['head'] for r in A), len(A))} | unanswerable refused {fmt(sum(r['ab'] for r in U), len(U))}")
        print("threshold sweep at min_coverage 0.5")
        for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            res = score_all(RAG(docs, threshold=t), qs, dbi, True)
            A = [r for q, r in zip(qs, res) if not q["imp"]]; U = [r for q, r in zip(qs, res) if q["imp"]]
            print(f"  thr={t}: answerable correct {fmt(sum(r['head'] for r in A), len(A))} | unanswerable refused {fmt(sum(r['ab'] for r in U), len(U))}")
        return
    t0 = time.time()
    fixed = RAG(docs); res = score_all(fixed, qs, dbi, True)
    print("FROZEN RUN, defaults unchanged (threshold 0.30, window 3, min_coverage 0.5)")
    A, U = summarize("fixed RAG", qs, res)
    print(f"  fixed done in {time.time()-t0:.0f}s")
    t0 = time.time()
    bres = score_all(BrokenRAG(docs), qs, dbi, False)
    summarize("BrokenRAG", qs, bres)
    print(f"  broken done in {time.time()-t0:.0f}s")
    # categories
    cats = {k: [] for k in ("refused_low_score", "refused_low_coverage", "wrong_paragraph", "right_paragraph_wrong_sentence", "answered_from_own_paragraph", "answered_from_other_paragraph")}
    for q, r in zip(qs, res):
        if q["imp"]:
            if not r["ab"]:
                cats["answered_from_own_paragraph" if r["cit"] == q["own"] else "answered_from_other_paragraph"].append((q, r))
        elif not r["head"]:
            if r["ab"]:
                cats["refused_low_score" if r["score"] < fixed.threshold else "refused_low_coverage"].append((q, r))
            elif contains(dbi[r["cit"]], q["golds"]):
                cats["right_paragraph_wrong_sentence"].append((q, r))
            else:
                cats["wrong_paragraph"].append((q, r))
    print("\nFAILURE CATEGORIES (fixed RAG)")
    for k, v in cats.items():
        print(f"{k}: {len(v)}")
    for k, v in cats.items():
        print(f"\n--- {k}: first 5 examples")
        for q, r in v[:5]:
            print(f"  Q: {q['q']}\n    gold: {q['golds'][:2] if not q['imp'] else '(unanswerable)'}\n    got: {r['ans'][:200]!r} cited={r['cit']} score={r['score']:.2f} cov={r['cov']:.2f}")
    # spot check sample
    pool = [(q, r) for q, r in zip(qs, res) if q["imp"] and not r["ab"]]
    samp = random.Random(20261103).sample(pool, min(30, len(pool)))
    print(f"\nSPOT CHECK SAMPLE: {len(samp)} of {len(pool)} unanswerable questions that fixed RAG answered")
    for i, (q, r) in enumerate(samp, 1):
        print(f"[{i}] {q['id']} Q: {q['q']}\n     own={q['own']} cited={r['cit']}\n     returned: {r['ans']}")
main()
