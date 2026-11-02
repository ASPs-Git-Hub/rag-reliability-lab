"""Tiny retrieval-augmented answerer, stdlib only. Synthetic FAQ for a fictional grocer.

Two implementations of the same interface:
  BrokenRAG: the failure modes seen in real prototypes
    1 case/punctuation-sensitive tokens ("days." never matches "days")
    2 fixed-width chunks that cut sentences in half
    3 always answers with the top chunk, never abstains
    4 no citations
  RAG: normalised tokens, sentence-aware chunks, abstains below a score threshold, cites sources.
No LLM is called: answers are extractive, so every output sentence is traceable to a source.
"""
import math
import re
from collections import Counter

STOP = {"a", "an", "the", "is", "are", "do", "does", "you", "i", "can", "to", "of", "in", "on",
        "for", "if", "we", "it", "how", "what", "my", "or", "and", "be", "by", "at", "me", "your",
        "with", "after", "up", "that", "this", "an", "from", "will"}


# Generic question words that say nothing about the topic. Chosen as a fixed list, not per question.
GENERIC = {"long", "much", "many", "take", "offer", "have", "get", "need", "want", "time", "open",
           "available", "work", "use", "possible", "way", "do", "does", "when", "where", "who", "which"}


def load_docs(text):
    """Parse '[id] paragraph' blocks into [(doc_id, text)]."""
    docs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        m = re.match(r"\[(\w+)\]\s*(.*)", block.strip(), re.S)
        if m:
            docs.append((m.group(1), " ".join(m.group(2).split())))
    return docs


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def norm_tokens(text):
    """Toy normaliser: lowercase, strip dots, crude plural stripping ("plus" becomes "plu"). Fine for an 8-paragraph demo only."""
    out = []
    for w in re.findall(r"[a-z0-9.]+", text.lower()):
        w = w.strip(".")
        if not w or w in STOP:
            continue
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def raw_tokens(text):  # BrokenRAG: no lowercasing, punctuation kept
    return text.split()


class _Index:
    def __init__(self, chunks, tokenize):
        self.chunks, self.tok = chunks, tokenize
        self.vecs = [Counter(tokenize(c["text"])) for c in chunks]
        df = Counter(t for v in self.vecs for t in v)
        n = len(chunks)
        self.idf = {t: math.log((n + 1) / (d + 0.5)) + 1 for t, d in df.items()}

    def _w(self, counts):
        return {t: c * self.idf.get(t, 0) for t, c in counts.items()}

    def search(self, query, k=3):
        q = self._w(Counter(self.tok(query)))
        qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scored = []
        for ch, vec in zip(self.chunks, self.vecs):
            d = self._w(vec)
            dn = math.sqrt(sum(v * v for v in d.values())) or 1.0
            s = sum(q[t] * d.get(t, 0) for t in q) / (qn * dn)
            scored.append((s, ch))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


class BrokenRAG:
    def __init__(self, docs, width=70):
        chunks = []
        for did, text in docs:
            for i in range(0, len(text), width):
                chunks.append({"doc": did, "text": text[i:i + width]})
        self.index = _Index(chunks, raw_tokens)

    def answer(self, question):
        score, ch = self.index.search(question, 1)[0]
        return {"answer": ch["text"], "citations": [], "abstained": False, "score": score, "coverage": 1.0}


class RAG:
    def __init__(self, docs, threshold=0.30, window=3, min_coverage=0.5):  # 0.5: plateau of the pooled sweep, evidence/run5
        if not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer")
        self.min_coverage = min_coverage
        chunks = []
        for did, text in docs:
            sents = sentences(text)
            for i in range(0, len(sents), window):  # non-overlapping, so no tiny trailing chunk
                chunks.append({"doc": did, "text": " ".join(sents[i:i + window]),
                               "sents": sents[i:i + window]})
        # topic words from the doc id are indexed twice (title boost), a standard retrieval fix
        self.index = _Index(chunks, lambda t: norm_tokens(t))
        for ch, vec in zip(self.index.chunks, self.index.vecs):
            for tok in norm_tokens(ch["doc"]):
                vec[tok] += 2
        df = Counter(t for v in self.index.vecs for t in v)
        n = len(self.index.chunks)
        self.index.idf = {t: math.log((n + 1) / (d + 0.5)) + 1 for t, d in df.items()}
        self.threshold = threshold

    def answer(self, question):
        no_answer = {"answer": "I could not find that in the FAQ.", "citations": [], "abstained": True}
        hits = self.index.search(question, 3)
        if not hits:  # empty corpus
            return {**no_answer, "score": 0.0, "coverage": 0.0}
        score, ch = hits[0]
        # Answerability gate: topic similarity is not enough. The question's own content words
        # must be found in the retrieved text, otherwise ("return address") we abstain.
        content = {t for t in norm_tokens(question) if t not in GENERIC}
        have = set(norm_tokens(ch["text"])) | set(norm_tokens(ch["doc"]))
        coverage = (len(content & have) / len(content)) if content else 0.0
        if score < self.threshold or coverage < self.min_coverage:
            return {**no_answer, "score": score, "coverage": coverage}
        q = set(norm_tokens(question))
        best = sorted(ch["sents"], key=lambda s: -len(q & set(norm_tokens(s))))
        picked = [s for s in best[:2] if q & set(norm_tokens(s))] or best[:1]
        return {"answer": " ".join(picked), "citations": [ch["doc"]], "abstained": False,
                "score": score, "coverage": coverage}
