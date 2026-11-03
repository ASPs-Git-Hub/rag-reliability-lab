# RAG Reliability Lab

**Independent technical demonstration. The tests use made-up cases on purpose; the main results use real public data (see `real-data/`). Not client work.**

Status: **CI-PASSING**, independent evaluation pending. Last full remote CI run: commit `bc9e4f6`, [run 41601092474](https://github.com/ASPs-Git-Hub/rag-reliability-lab/actions/runs/41601092474), Python 3.10 and 3.12: unit tests and the planted-defect check. I also ran everything locally on 2026-11-03. No outside person has checked the results. The confidentiality gate (`check-confidential.sh`) was run locally by me, not in CI. Every made-up test set was written by me with an AI model, so none is independent human labelling.

Details of what I ran: I tested the first version myself on 2026-11-02 (Python 3.14, standard library only). Two independent AI reviewers (ChatGPT and Claude Fable) then reviewed it and I corrected what they found. No human has reviewed the engineering. Nothing is deployed.

## Real-data results
Full write-up, rules and raw output: [`real-data/README.md`](real-data/README.md). Data: SQuAD 2.0 (Wikipedia text, public), first 20 articles: 3,068 questions with an answer and 3,179 questions that look answerable but are not. I wrote the rules before the run and ran the test once.

| Version | Questions with an answer, answered correctly | Questions without an answer, refused |
|---|---|---|
| v1 (`RAG`) | 42.4% | 59.1% |
| v2 (`RAGv2`, refuse below 0.50) | 33.5% | 75.0% |
| v2, refuse below 0.40 | 49.1% | 60.2% |

- v1 is much better than the broken version, but it is wrong a lot: it refuses almost half the questions that have an answer.
- v2 (`src/rag2.py`) is a new version I built after seeing why v1 failed. It was fitted on other articles, not on the test ones.
- The gain is modest, not a fix. v2 does worse on the old grocery FAQ: it refuses almost everything there.
- The "refused" column is only for tricky questions made to fool programs. Real customers' questions are probably easier.

## Problem
RAG prototypes that work in a notebook usually fail in two ways. They answer wrongly, or invent an answer, when the knowledge base has none. And they break at deployment (typically Hugging Face Spaces). I reproduced both, wrote a fix, and measured how well the fix works.

## What is in it
- `src/rag.py`: `BrokenRAG` (case and punctuation sensitive tokens, fixed-width chunks that cut sentences, never abstains, no citations) and `RAG` (normalised tokens, sentence-aware chunks, title boost, an answerability gate, source citation). Answers are extractive and no LLM is called, so every sentence can be traced to its source.
- `src/evaluate.py`: four question sets, counts with Wilson 95% intervals, an answerability sweep.
- `src/doctor.py`: static deployment checks for the usual Colab-to-Spaces causes (unpinned or loosely pinned requirements, wrong host or port, hard-coded secrets, Colab-only code, missing input files, README front matter). Reads files only.
- `fixtures/colab_style` (breaks every check) and `fixtures/spaces_ready` (clean).
- `src/rag2.py`: `RAGv2`, the second version (sentence search with BM25, question-type bonus, a learned "should I answer?" gate). The old `RAG` and `BrokenRAG` are unchanged.
- `tests/test_lab.py` (27 original tests) and `tests/test_rag2.py` (33 new tests): 60 tests in total, all pass. `scripts/mutate.py`: 41 planted defects (19 old, 22 for v2), all 41 caught.
Run: `python3 -m unittest discover -s tests`, `python3 src/evaluate.py [--verbose|--sweep]`, `python3 scripts/mutate.py`.

## Results (evidence in `evidence/`, run 2026-11-02)
Answerable questions answered correctly / on-topic or off-topic unanswerable questions correctly refused (counts, then Wilson 95% interval):

| Set | Fixed: answerable correct | Fixed: unanswerable refused | Broken: answerable / refused |
|---|---|---|---|
| Tuned (14 q), used while building | 10/10 | 4/4 | 1/10, 0/4 |
| Second set (8 q) | 6/6 | 2/2 | 1/6, 0/2 |
| Frozen set 3 (20 q) | 7/10 | 7/10 | 5/10, 0/10 |
| Frozen set 4 (17 q), the only clean test | 8/8 (95% CI 68-100%) | 6/9 (95% CI 35-88%) | 3/8, 0/9 |

Only set 4 is a clean test, because it was written and committed before I chose the default threshold. Sets 3 and 4 were committed on their own before the code they test (see git history). The tuned and second sets, and set 3 after the sweep, are optimistic.
- `evidence/run1-fixed-v0-failures.txt`: the first fixed version failed two questions, found by the harness.
- `evidence/run3-set3-before-changes.txt`: before the answerability gate, the fixed system refused only 3 of 10 on-topic unanswerable questions in set 3. The earlier README figure of "1.00 correct abstention" was true only for off-topic questions.
- `evidence/run5-coverage-sweep.txt`: the gate's precision/recall trade-off. Strict thresholds refuse more traps, and also refuse more valid paraphrases (at 0.6, 16/26 answerable pooled; at 0.5, 23/26).
- `evidence/mutation-run.txt`: 19 of 19 planted defects detected (the first version; the current check catches 41 of 41, raw output in `real-data/evidence/v2-06-mutation.txt`).
- `evidence/doctor-run.txt`: 7 findings on the Colab-style fixture (all six check codes), 0 on the Spaces-ready fixture.

## What this does NOT prove
- The sets are handwritten and tiny, and the intervals are wide. These numbers show the method. They are not a benchmark and not a population estimate.
- The gate is lexical, and known failures remain. It still answers some on-topic traps (for example "What is the delivery cost for orders under 50 dollars with standard delivery?") and refuses some valid paraphrases that use words absent from the FAQ ("shipping", "helpline", "allergy"). A real system needs semantic retrieval plus an answerability or verification step and a much larger evaluation set.
- The answer grader is crude: a phrase match with a three-word negation guard.
- Retrieval is TF-IDF over eight paragraphs: nothing here speaks to embeddings, large corpora, latency or cost.
- The doctor is static and regex-based: it cannot see runtime errors, so a clean report does not prove a Space will start. No deployment was performed or verified.
- No LLM and no real users.
- Real-data limits: 20 Wikipedia articles are not a business's documents. The scoring is a text match, so a correct answer worded differently counts as wrong. The "unanswerable" questions were written to fool programs. The hand check of 30 answers was read by one person, and not repeated for v2. The v2 gate was fitted on Wikipedia and fails on the FAQ. I ran the test once and nobody else has reproduced it.
