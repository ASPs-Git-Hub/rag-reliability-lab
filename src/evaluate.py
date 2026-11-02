"""Evaluation harness. Three question sets, handwritten and tiny: a DEMONSTRATION, not a benchmark.

  TUNED    used while building the system (so scores on it are optimistic)
  HELDOUT  written after the first tuning round (second set, 8 questions)
  SET3     data/set3_frozen.json: 20 questions, 10 on-topic unanswerable, committed on its own BEFORE the
           answerability gate was written (see git history). The default threshold was then picked from a
           sweep that included SET3, so SET3 scores are optimistic too
  SET4     data/set4_frozen.json: 17 questions committed BEFORE the default was chosen: the only clean test

Metrics are reported as counts with Wilson 95% intervals. Grading rules (crude, documented):
  answerable  correct = not abstained AND expected phrase in the answer AND no negation just before it
  unanswerable correct = abstained
  citation    counted only on answerable questions: the correct source document is cited
"""
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from rag import BrokenRAG, RAG, load_docs  # noqa: E402

HERE = os.path.dirname(os.path.dirname(__file__))

TUNED = [
    ("How long does standard delivery take?", "2 to 4 business days", "delivery"),
    ("How much is express delivery?", "9.99", "delivery"),
    ("Is delivery free?", "over 50 dollars", "delivery"),
    ("Can I return opened items?", "14 days", "returns"),
    ("How long do refunds take?", "5 to 7 business days", "refunds"),
    ("How long is the password reset link valid?", "60 minutes", "password"),
    ("Can I change my delivery slot?", "2 hours", "slots"),
    ("What are the support hours?", "8 am to 10 pm", "support"),
    ("Do you guarantee a nut-free warehouse?", "cannot guarantee", "allergens"),
    ("What happens if an item is out of stock?", "substitute", "substitutions"),
    ("Do you deliver to Antarctica?", None, None),
    ("Who is the CEO of the company?", None, None),
    ("Can I pay with cryptocurrency?", None, None),
    ("Do you sell gift cards?", None, None),
]

HELDOUT = [
    ("How many days to get a refund after returning something?", "5 to 7 business days", "refunds"),
    ("Is there a phone number for support?", "phone support is not offered", "support"),
    ("When does the password reset link expire?", "60 minutes", "password"),
    ("Up to when can I edit my delivery time slot?", "2 hours", "slots"),
    ("Can I turn off product substitutions?", "disable", "substitutions"),
    ("Do you list allergens on products?", "product page", "allergens"),
    ("Do you offer a loyalty program?", None, None),
    ("What is your return address in Canada?", None, None),
]


def load_set3():
    with open(os.path.join(HERE, "data", "set3_frozen.json"), encoding="utf-8") as f:
        return [(x["q"], x["expect"], x["doc"]) for x in json.load(f)["questions"]]


def load_set4():
    with open(os.path.join(HERE, "data", "set4_frozen.json"), encoding="utf-8") as f:
        return [(x["q"], x["expect"], x["doc"]) for x in json.load(f)["questions"]]


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def phrase_ok(answer, phrase):
    a = answer.lower()
    if phrase not in a:
        return False
    # negation within three words before the phrase ("not 30 days"); a wider window mis-graded a correct answer
    if re.search(r"\b(not|never|no|cannot)\s+(?:\w+\s+){0,3}" + re.escape(phrase), a) and not re.search(r"\b(not|cannot)\b", phrase):
        return False  # "not <expected phrase>" is a contradiction, not a match
    return True


def run(system, golden):
    rows = []
    for q, phrase, doc in golden:
        r = system.answer(q)
        if phrase is None:
            ok, cited = r["abstained"], None
        else:
            ok = (not r["abstained"]) and phrase_ok(r["answer"], phrase)
            cited = (not r["abstained"]) and doc in r["citations"]
        rows.append({"q": q, "answerable": phrase is not None, "ok": ok, "cited": cited,
                     "answer": r["answer"], "score": round(r["score"], 3), "coverage": round(r.get("coverage", 0), 3)})
    ans = [x for x in rows if x["answerable"]]
    una = [x for x in rows if not x["answerable"]]
    return {
        "answerable_correct": (sum(x["ok"] for x in ans), len(ans)),
        "unanswerable_abstained": (sum(x["ok"] for x in una), len(una)),
        "citations_correct": (sum(bool(x["cited"]) for x in ans), len(ans)),
        "rows": rows,
    }


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n}" if n == 0 else f"{k}/{n} (95% CI {lo:.0%}-{hi:.0%})"


def sweep(docs):
    """Precision/recall trade-off of the answerability gate, on all 42 questions pooled."""
    allq = TUNED + HELDOUT + load_set3()
    print("min_coverage  answerable_correct  unanswerable_abstained   (pooled: tuned + second + frozen set 3)")
    for cov in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        m = run(RAG(docs, min_coverage=cov), allq)
        print(f"{cov:>12}  {fmt(*m['answerable_correct']):<32} {fmt(*m['unanswerable_abstained'])}")


def main():
    with open(os.path.join(HERE, "data", "faq.txt"), encoding="utf-8") as f:
        docs = load_docs(f.read())
    if "--sweep" in sys.argv:
        return sweep(docs)
    sets = (("tuned (14 q)", TUNED), ("second set (8 q)", HELDOUT), ("frozen set 3 (20 q)", load_set3()),
            ("frozen set 4 (17 q, clean test)", load_set4()))
    for label, golden in sets:
        for name, system in (("broken", BrokenRAG(docs)), ("fixed", RAG(docs))):
            m = run(system, golden)
            print(f"[{label}] {name}: answerable correct {fmt(*m['answerable_correct'])}; "
                  f"unanswerable abstained {fmt(*m['unanswerable_abstained'])}; "
                  f"citations correct {fmt(*m['citations_correct'])}")
            if "--verbose" in sys.argv and name == "fixed":
                for r in m["rows"]:
                    if not r["ok"]:
                        print("   MISS", r["q"], "->", r["answer"][:60], f"(score {r['score']}, coverage {r['coverage']})")


if __name__ == "__main__":
    main()
