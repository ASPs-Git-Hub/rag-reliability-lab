"""RAGv2: BM25 sentence retrieval plus a feature-based abstain rule. Standard library only, no model download.

Why it exists: the first fixed RAG (rag.RAG) judged by shared words with a hand-set limit. On real data
(SQuAD 2.0) that cannot separate paraphrased real questions from the crowd-written trick questions.
This version changes four things:
  1 retrieval: BM25 over sentences, mixed with BM25 over the whole paragraph (title words added), and a
    small bonus when the sentence holds what the question type needs (a year for "when", a number for
    "how many", a capitalised name for "who" / "where")
  2 answer: the single best sentence is returned (no padding with a second sentence)
  3 abstain rule: a logistic score over ten features that all measure how well the best sentence backs the
    question (rare-word coverage, word-pair coverage, missing names, missing answer type, margin to the
    second-best sentence, ...)
  4 the weights are fitted on a separate tuning set and frozen; see real-data/PROTOCOL-v2.md
rag.RAG and rag.BrokenRAG are untouched.
"""
import math
import re
from collections import Counter, defaultdict

from rag import sentences

STOP2 = set("""a an the is are was were be been being am do does did done doing have has had having i you he she it we they
me him her us them my your his its our their this that these those of in on at by for with from to into onto over under
about as than then so and or but if not no nor also too very can could will would shall should may might must there here
what which who whom whose when where why how many much more most other another some any each both such same own up out
off after before during between through against s t""".split())

MONTHS = set("""january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec""".split())
NUMWORDS = set("""one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen
eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion dozen half
first second third fourth fifth""".split())

FEATURES = ["bias", "cov_sent", "cov_par", "score", "margin", "bigram_sent", "bigram_par",
            "ent_miss", "type_miss", "rare_miss", "n_terms", "miss_par_cnt", "num_miss", "ent_miss_sent",
            "compact", "matched_rare", "miss_sent_rare", "cov_sent_sq"]


def stem(w):
    """Stemming-lite: a few suffix rules, nothing clever. 'countries' and 'country' meet; 'class' stays 'class'."""
    n = len(w)
    if n > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if n > 5 and w.endswith("ing"):
        return w[:-3]
    if n > 4 and w.endswith("ed"):
        return w[:-2]
    if n > 4 and w.endswith(("ses", "xes", "ches", "shes")):
        return w[:-2]
    if n > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def tokens2(text):
    text = text.lower().replace("'s", "").replace("’s", "")
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text) if w not in STOP2]


def question_type(question):
    """'num', 'date', 'proper' or None. Fixed rules, written before any data was looked at."""
    q = question.lower()
    if re.search(r"\bhow (many|much|old|large|big|high|long ago)\b|\bwhat (number|percent|percentage|amount)\b", q):
        return "num"
    if re.search(r"\bwhen\b|\bwhat (year|date|century|decade|month|day)\b|\bin which (year|century)\b", q):
        return "date"
    if re.search(r"\bwho(m|se)?\b|\bwhere\b|\bwhat (country|city|state|town|nation|person|people|region|river)\b", q):
        return "proper"
    return None


def type_evidence(qtype, sentence, question):
    """True if the sentence contains something of the type the question asks for."""
    if qtype is None:
        return True
    low = re.findall(r"[a-z0-9]+", sentence.lower())
    if qtype == "num":
        return any(w.isdigit() or w in NUMWORDS for w in low)
    if qtype == "date":
        return any((w.isdigit() and len(w) >= 3) or w in MONTHS or w in ("century", "decade") for w in low) or bool(re.search(r"\d{1,2}(st|nd|rd|th)", sentence.lower()))
    qwords = set(re.findall(r"[a-z]+", question.lower()))
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", sentence)
    for i, w in enumerate(words):
        if i > 0 and w[0].isupper() and w.lower() not in qwords and w.lower() not in STOP2:
            return True
    return False


def question_entities(question):
    """Capitalised words that are not the first word and not question words, stemmed."""
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", question)
    return [stem(w.lower().replace("'s", "")) for i, w in enumerate(words)
            if i > 0 and w[0].isupper() and w.lower() not in STOP2]


class _Bm25:
    def __init__(self, docs_tokens, k1, b):
        self.k1, self.b, self.n = k1, b, len(docs_tokens)
        self.len = [len(d) for d in docs_tokens]
        self.avg = (sum(self.len) / self.n) if self.n else 1.0
        self.avg = self.avg or 1.0
        self.post = defaultdict(list)
        for i, d in enumerate(docs_tokens):
            for t, c in Counter(d).items():
                self.post[t].append((i, c))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}
        self.idf_unseen = math.log(1 + (self.n + 0.5) / 0.5)

    def idf_of(self, t):
        return self.idf.get(t, self.idf_unseen)

    def scores(self, qterms):
        out = defaultdict(float)
        for t in qterms:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in self.post[t]:
                norm = 1 - self.b + self.b * self.len[i] / self.avg
                out[i] += idf * c * (self.k1 + 1) / (c + self.k1 * norm)
        return out


DEFAULT_RETRIEVAL = {"k1": 0.9, "b": 0.5, "w_par": 0.6, "w_type": 0.1}  # chosen on the tuning set, protocol-v2
# Placeholder gate: used only until fitted weights are frozen below.
UNFITTED_GATE = {"mean": [0.0] * len(FEATURES), "std": [1.0] * len(FEATURES), "w": [0.0] * len(FEATURES), "threshold": 0.5}


class RAGv2:
    def __init__(self, docs, retrieval=None, gate=None):
        self.r = {**DEFAULT_RETRIEVAL, **(retrieval or {})}
        self.gate = gate or FROZEN_GATE
        self.docs = docs
        self.sents = []          # (doc_index, sentence_text)
        self.sent_tokens = []
        self.par_tokens = []
        self.par_seq = []
        self.titles = []
        for di, (did, text) in enumerate(docs):
            title = re.sub(r"_p\d+$", "", did).replace("_", " ")
            ptoks = tokens2(text)
            ttoks = tokens2(title)
            self.titles.append(ttoks)
            self.par_seq.append(ptoks)
            self.par_tokens.append(ptoks + ttoks + ttoks)
            for s in sentences(text):
                self.sents.append((di, s))
                self.sent_tokens.append(tokens2(s))
        self.sidx = _Bm25(self.sent_tokens, self.r["k1"], self.r["b"])
        self.pidx = _Bm25(self.par_tokens, self.r["k1"], self.r["b"])
        self.par_sets = [set(p) | set(t) for p, t in zip(self.par_seq, self.titles)]

    # ---- retrieval -------------------------------------------------------------
    def rank(self, question, topn=2):
        """Returns [(combined_score, sentence_index)] best first, and the query term list."""
        qterms = list(dict.fromkeys(tokens2(question)))
        ss = self.sidx.scores(qterms)
        ps = self.pidx.scores(qterms)
        s_max = (self.r["k1"] + 1) * sum(self.sidx.idf_of(t) for t in qterms) or 1.0
        p_max = (self.r["k1"] + 1) * sum(self.pidx.idf_of(t) for t in qterms) or 1.0
        qtype = question_type(question)
        scored = []
        for i, sv in ss.items():
            di, text = self.sents[i]
            sc = sv / s_max + self.r["w_par"] * ps.get(di, 0.0) / p_max
            if qtype and type_evidence(qtype, text, question):
                sc += self.r["w_type"]
            scored.append((sc, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[:topn], qterms

    # ---- gate features -----------------------------------------------------------
    def features(self, question, ranked, qterms):
        (sc1, i1) = ranked[0]
        sc2 = ranked[1][0] if len(ranked) > 1 else 0.0
        di, text = self.sents[i1]
        stoks = set(self.sent_tokens[i1])
        ptoks = self.par_sets[di]
        idf = self.sidx.idf_of
        total = sum(idf(t) for t in qterms) or 1.0
        cov_sent = sum(idf(t) for t in qterms if t in stoks) / total
        cov_par = sum(idf(t) for t in qterms if t in ptoks) / total
        qseq = [t for t in tokens2(question)]
        pairs = list(zip(qseq, qseq[1:]))
        sseq = self.sent_tokens[i1]
        sp = set(zip(sseq, sseq[1:]))
        pp = set(zip(self.par_seq[di], self.par_seq[di][1:]))
        bigram_sent = (sum(p in sp for p in pairs) / len(pairs)) if pairs else cov_sent
        bigram_par = (sum(p in pp for p in pairs) / len(pairs)) if pairs else cov_par
        ents = question_entities(question)
        ent_miss = (sum(e not in ptoks for e in ents) / len(ents)) if ents else 0.0
        qtype = question_type(question)
        type_miss = 0.0 if type_evidence(qtype, text, question) else 1.0
        miss_idf = [idf(t) for t in qterms if t not in ptoks]
        top = math.log(1 + (self.sidx.n + 0.5) / 0.5) or 1.0
        rare_miss = (max(miss_idf) / top) if miss_idf else 0.0
        margin = (sc1 - sc2) / sc1 if sc1 > 0 else 0.0
        n_terms = min(len(qterms), 10) / 10
        miss_par_cnt = min(sum(t not in ptoks for t in qterms), 5) / 5
        nums = [t for t in qterms if t.isdigit()]
        num_miss = (sum(t not in ptoks for t in nums) / len(nums)) if nums else 0.0
        ent_miss_sent = (sum(e not in stoks for e in ents) / len(ents)) if ents else 0.0
        # compactness: how tightly the matched question words sit together in the sentence
        pos = [k for k, t in enumerate(sseq) if t in set(qterms)]
        matched_n = len({sseq[k] for k in pos})
        compact = 1.0
        if matched_n >= 2:
            best_span = min(pos[b] - pos[a] + 1 for a in range(len(pos)) for b in range(a, len(pos))
                            if len({sseq[k] for k in pos[a:b + 1]}) == matched_n)
            compact = matched_n / best_span
        matched = [idf(t) for t in qterms if t in stoks]
        matched_rare = (max(matched) / top) if matched else 0.0
        miss_s = [idf(t) for t in qterms if t not in stoks]
        miss_sent_rare = (max(miss_s) / top) if miss_s else 0.0
        return [1.0, cov_sent, cov_par, sc1, margin, bigram_sent, bigram_par, ent_miss, type_miss, rare_miss,
                n_terms, miss_par_cnt, num_miss, ent_miss_sent, compact, matched_rare, miss_sent_rare, cov_sent ** 2]

    def probability(self, x):
        g = self.gate
        z = g["w"][0]
        for j in range(1, len(x)):
            z += g["w"][j] * (x[j] - g["mean"][j]) / g["std"][j]
        return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))

    def candidate(self, question):
        """The retrieval step on its own: (sentence, doc_id, feature_vector) or None."""
        ranked, qterms = self.rank(question)
        if not ranked:
            return None
        di, text = self.sents[ranked[0][1]]
        return text, self.docs[di][0], self.features(question, ranked, qterms)

    def answer(self, question):
        no_answer = {"answer": "I could not find that in the FAQ.", "citations": [], "abstained": True}
        cand = self.candidate(question)
        if cand is None:
            return {**no_answer, "score": 0.0, "coverage": 0.0}
        text, did, x = cand
        p = self.probability(x)
        if p < self.gate["threshold"]:
            return {**no_answer, "score": p, "coverage": x[1], "candidate": text, "candidate_doc": did}
        return {"answer": text, "citations": [did], "abstained": False, "score": p, "coverage": x[1],
                "candidate": text, "candidate_doc": did}


# Frozen on the tuning set (SQuAD 2.0 dev articles 21 to 35), see real-data/evidence/v2-01-tuning.txt and PROTOCOL-v2.md.
# Order of the numbers follows FEATURES. Nothing here was fitted on the test articles.
FROZEN_GATE = {
    "w": [-0.356137, 0.316744, -0.306526, 0.228128, 0.0349334, -0.0134736, 0.102531, -0.351247, -0.0976163, 0.159319, 0.257737, 0.0434794, -0.315596, 0.178274, -0.0333393, 0.095641, -0.363167, 0.194608],
    "mean": [0, 0.676762, 0.747031, 0.677201, 0.242716, 0.337042, 0.361142, 0.0754742, 0.0563829, 0.550506, 0.526215, 0.253746, 0.0207803, 0.154119, 0.697229, 0.73719, 0.615228, 0.502068],
    "std": [1, 0.209909, 0.198213, 0.19423, 0.165301, 0.283867, 0.284185, 0.232251, 0.23066, 0.356946, 0.173763, 0.214774, 0.140137, 0.32143, 0.276164, 0.106247, 0.316355, 0.285301],
    "threshold": 0.5,
}
