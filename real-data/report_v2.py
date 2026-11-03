"""The scored v2 report. Called by run_eval_v2.py --split test (the frozen run) or --split tune --report (a dry run on tuning data).
Nothing here fits or chooses anything: it applies the frozen v1 and the frozen v2 to the same questions."""
import hashlib, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from rag import RAG  # noqa: E402
import rag2  # noqa: E402
from rag2 import RAGv2  # noqa: E402
import run_eval_v2 as R  # noqa: E402

def sha(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    except OSError:
        return "(file absent)"

def run(docs, qs):
    dbi = dict(docs)
    print("files in this run (first 16 hex of sha256):")
    for rel in ("../src/rag.py", "../src/rag2.py", "PROTOCOL-v2.md", "run_eval_v2.py", "report_v2.py"):
        print(f"  {rel}: {sha(os.path.join(HERE, rel))}")
    print(f"v2 frozen gate threshold {rag2.FROZEN_GATE['threshold']}; retrieval {rag2.DEFAULT_RETRIEVAL}")
    A = [i for i, q in enumerate(qs) if not q["imp"]]; U = [i for i, q in enumerate(qs) if q["imp"]]
    t0 = time_start = __import__("time").time()
    v1 = RAG(docs)
    r1 = [v1.answer(q["q"]) for q in qs]
    print(f"v1 done in {__import__('time').time()-t0:.0f}s")
    v2 = RAGv2(docs)
    cands = [v2.candidate(q["q"]) for q in qs]
    probs = [(v2.probability(c[2]) if c else 0.0) for c in cands]
    thr = rag2.FROZEN_GATE["threshold"]

    def outcome2(i, t):
        c = cands[i]
        return (c is None) or probs[i] < t   # abstained

    def head1(i):
        r = r1[i]
        return (not r["abstained"]) and R.contains(r["answer"], qs[i]["golds"])
    def cand_ok(i):
        return cands[i] is not None and R.contains(cands[i][0], qs[i]["golds"])
    def head2(i, t=thr):
        return (not outcome2(i, t)) and cand_ok(i)
    def loose2(i, t=thr):
        return (not outcome2(i, t)) and R.contains(dbi[cands[i][1]], qs[i]["golds"])
    def loose1(i):
        r = r1[i]
        return (not r["abstained"]) and R.contains(dbi[r["citations"][0]], qs[i]["golds"])

    print("\nHEADLINE (same frozen questions, same scoring rule as PROTOCOL.md)")
    rows = {}
    for name, hf, lf, abf in (("v1 (RAG, before)", head1, loose1, lambda i: r1[i]["abstained"]),
                              ("v2 (RAGv2, after)", head2, loose2, lambda i: outcome2(i, thr))):
        ac = sum(hf(i) for i in A); lo = sum(lf(i) for i in A); ur = sum(abf(i) for i in U); ar = sum(abf(i) for i in A)
        rows[name] = (ac, ur)
        print(f"{name}: answerable correct {R.fmt(ac, len(A))}")
        print(f"{name}: answerable correct, looser (cited paragraph holds the answer) {R.fmt(lo, len(A))}")
        print(f"{name}: answerable refused {R.fmt(ar, len(A))}")
        print(f"{name}: unanswerable correctly refused {R.fmt(ur, len(U))}")
        print(f"{name}: mean of the two rates (the pre-declared objective) {(ac/len(A)+ur/len(U))/2:.1%}")
    # consistency with the stored frozen v1 run
    print(f"CHECK v1 reproduces the frozen run in evidence/01-frozen-run.txt (1300 answerable correct, 1879 refused): "
          f"{rows['v1 (RAG, before)'][0]} and {rows['v1 (RAG, before)'][1]} (only meaningful on the test split)")

    # paired view
    both = sum(head1(i) and head2(i) for i in A); o1 = sum(head1(i) and not head2(i) for i in A)
    o2 = sum(head2(i) and not head1(i) for i in A); nei = len(A) - both - o1 - o2
    print(f"\nPAIRED, answerable questions: both right {both}; only v1 right {o1}; only v2 right {o2}; neither {nei}")
    bu = sum(r1[i]["abstained"] and outcome2(i, thr) for i in U); u1 = sum(r1[i]["abstained"] and not outcome2(i, thr) for i in U)
    u2 = sum(outcome2(i, thr) and not r1[i]["abstained"] for i in U); un = len(U) - bu - u1 - u2
    print(f"PAIRED, unanswerable questions: both refuse {bu}; only v1 refuses {u1}; only v2 refuses {u2}; neither {un}")

    # categories
    def cats1():
        c = dict.fromkeys(("correct", "refused_answerable", "wrong_paragraph", "right_paragraph_wrong_sentence", "answered_from_own_paragraph", "answered_from_other_paragraph", "correct_refusal"), 0)
        for i, q in enumerate(qs):
            r = r1[i]
            if q["imp"]:
                if r["abstained"]: c["correct_refusal"] += 1
                else: c["answered_from_own_paragraph" if r["citations"][0] == q["own"] else "answered_from_other_paragraph"] += 1
            elif head1(i): c["correct"] += 1
            elif r["abstained"]: c["refused_answerable"] += 1
            elif R.contains(dbi[r["citations"][0]], q["golds"]): c["right_paragraph_wrong_sentence"] += 1
            else: c["wrong_paragraph"] += 1
        return c
    def cats2():
        c = dict.fromkeys(("correct", "refused_answerable", "wrong_paragraph", "right_paragraph_wrong_sentence", "answered_from_own_paragraph", "answered_from_other_paragraph", "correct_refusal", "refused_though_sentence_was_right", "refused_and_sentence_was_wrong"), 0)
        for i, q in enumerate(qs):
            ab = outcome2(i, thr)
            if q["imp"]:
                if ab: c["correct_refusal"] += 1
                else: c["answered_from_own_paragraph" if cands[i][1] == q["own"] else "answered_from_other_paragraph"] += 1
            elif head2(i): c["correct"] += 1
            elif ab:
                c["refused_answerable"] += 1
                c["refused_though_sentence_was_right" if cand_ok(i) else "refused_and_sentence_was_wrong"] += 1
            elif R.contains(dbi[cands[i][1]], q["golds"]): c["right_paragraph_wrong_sentence"] += 1
            else: c["wrong_paragraph"] += 1
        return c
    c1, c2 = cats1(), cats2()
    print("\nERROR CATEGORIES, before vs after (same questions; every question lands in exactly one row of its kind)")
    print(f"{'category':44s}{'v1 before':>10s}{'v2 after':>10s}")
    for k in ("correct", "refused_answerable", "wrong_paragraph", "right_paragraph_wrong_sentence", "correct_refusal", "answered_from_own_paragraph", "answered_from_other_paragraph"):
        tag = "answerable: " if k in ("correct", "refused_answerable", "wrong_paragraph", "right_paragraph_wrong_sentence") else "unanswerable: "
        print(f"{tag + k:44s}{c1[k]:>10d}{c2[k]:>10d}")
    print(f"{'  of v2 refusals of answerable: sentence right':44s}{'n/a':>10s}{c2['refused_though_sentence_was_right']:>10d}")
    print(f"{'  of v2 refusals of answerable: sentence wrong':44s}{'n/a':>10s}{c2['refused_and_sentence_was_wrong']:>10d}")
    print(f"(v1 sums: answerable {sum(c1[k] for k in ('correct','refused_answerable','wrong_paragraph','right_paragraph_wrong_sentence'))}, unanswerable {sum(c1[k] for k in ('correct_refusal','answered_from_own_paragraph','answered_from_other_paragraph'))}; "
          f"v2 sums: answerable {sum(c2[k] for k in ('correct','refused_answerable','wrong_paragraph','right_paragraph_wrong_sentence'))}, unanswerable {sum(c2[k] for k in ('correct_refusal','answered_from_own_paragraph','answered_from_other_paragraph'))})")

    # forced-answer ceiling of v2 retrieval
    forced = sum(cand_ok(i) for i in A)
    print(f"\nRETRIEVAL ALONE (v2, never refusing): top sentence holds a gold answer for {R.fmt(forced, len(A))}")

    # trade-off curve
    print("\nTRADE-OFF CURVE of the frozen v2 scorer on these questions (threshold varied, NOT re-chosen; descriptive, not a second result)")
    pts = []
    for t in [i / 20 for i in range(1, 20)]:
        ac = sum(head2(i, t) for i in A); ur = sum(outcome2(i, t) for i in U)
        pts.append((t, ac, ur))
        print(f"  thr={t:.2f}: answerable correct {ac/len(A):.1%} | unanswerable refused {ur/len(U):.1%} | mean {(ac/len(A)+ur/len(U))/2:.1%}")
    v1ac, v1ur = rows["v1 (RAG, before)"]
    best_ac = max((ac for t, ac, ur in pts if ur / len(U) >= v1ur / len(U)), default=None)
    best_ur = max((ur for t, ac, ur in pts if ac / len(A) >= v1ac / len(A)), default=None)
    if best_ac is not None:
        print(f"  at least v1's refusal rate ({v1ur/len(U):.1%}), the best v2 threshold on this grid answers {best_ac/len(A):.1%} correctly (v1: {v1ac/len(A):.1%})")
    if best_ur is not None:
        print(f"  at least v1's answerable rate ({v1ac/len(A):.1%}), the best v2 threshold on this grid refuses {best_ur/len(U):.1%} of unanswerable (v1: {v1ur/len(U):.1%})")

    # examples
    def show(title, idxs, n=6):
        print(f"\n--- {title}: {len(idxs)}; first {n}")
        for i in idxs[:n]:
            q = qs[i]; c = cands[i]
            print(f"  Q: {q['q']}\n    gold: {q['golds'][:2] if not q['imp'] else '(unanswerable)'}\n    v2 sentence: {c[0][:180]!r} cited={c[1]} p={probs[i]:.2f}")
    show("answerable, v2 answered wrongly (wrong paragraph)", [i for i in A if not outcome2(i, thr) and not cand_ok(i) and not R.contains(dbi[cands[i][1]], qs[i]["golds"])])
    show("answerable, v2 refused though its sentence was right", [i for i in A if outcome2(i, thr) and cand_ok(i)])
    show("unanswerable, v2 answered (from own paragraph)", [i for i in U if not outcome2(i, thr) and cands[i][1] == qs[i]["own"]])
    pool = [i for i in U if not outcome2(i, thr)]
    samp = random.Random(20261103).sample(pool, min(30, len(pool)))
    print(f"\nSPOT CHECK SAMPLE (not yet read by hand): {len(samp)} of {len(pool)} unanswerable questions v2 answered")
    for k, i in enumerate(samp, 1):
        print(f"[{k}] {qs[i]['id']} Q: {qs[i]['q']}\n     own={qs[i]['own']} cited={cands[i][1]}\n     returned: {cands[i][0]}")
