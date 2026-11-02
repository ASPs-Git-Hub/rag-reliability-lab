# RAG Reliability Lab

**Independent technical demonstration, synthetic data (a fictional grocer's FAQ). Not a client result.**

Status: BUILT_UNVERIFIED. Self-tested on 2026-11-02 (Python 3.14, standard library only), then reviewed by two independent AI reviewers (ChatGPT and Claude Fable) and corrected. No human engineering review. Nothing is deployed.

## Problem
RAG prototypes that work in a notebook fail in two ways: they answer wrongly or invent an answer when the knowledge base has none, and they break at deployment (typically Hugging Face Spaces). This lab reproduces both, shows a fix, and measures honestly how well the fix works.

## What is in it
- `src/rag.py`: `BrokenRAG` (case and punctuation sensitive tokens, fixed-width chunks that cut sentences, never abstains, no citations) and `RAG` (normalised tokens, sentence-aware chunks, title boost, an answerability gate, source citation). Answers are extractive and no LLM is called, so every sentence is traceable.
- `src/evaluate.py`: four question sets, counts with Wilson 95% intervals, an answerability sweep.
- `src/doctor.py`: static deployment checks for the usual Colab-to-Spaces causes (unpinned or loosely pinned requirements, wrong host or port, hard-coded secrets, Colab-only code, missing input files, README front matter). Reads files only.
- `fixtures/colab_style` (breaks every check) and `fixtures/spaces_ready` (clean).
- `tests/test_lab.py`: 27 tests. `scripts/mutate.py`: 19 planted defects, all caught.
Run: `python3 -m unittest discover -s tests`, `python3 src/evaluate.py [--verbose|--sweep]`, `python3 scripts/mutate.py`.

## Results (evidence in `evidence/`, run 2026-11-02)
Answerable questions answered correctly / on-topic or off-topic unanswerable questions correctly refused (counts, then Wilson 95% interval):

| Set | Fixed: answerable correct | Fixed: unanswerable refused | Broken: answerable / refused |
|---|---|---|---|
| Tuned (14 q), used while building | 10/10 | 4/4 | 1/10, 0/4 |
| Second set (8 q) | 6/6 | 2/2 | 1/6, 0/2 |
| Frozen set 3 (20 q) | 7/10 | 7/10 | 5/10, 0/10 |
| Frozen set 4 (17 q), the only clean test | 8/8 (95% CI 68-100%) | 6/9 (95% CI 35-88%) | 3/8, 0/9 |

Only set 4 is a clean test: it was written and committed before the default threshold was chosen. Sets 3 and 4 were committed on their own before the code that they test (see git history). The tuned and second sets, and set 3 after the sweep, are optimistic.
- `evidence/run1-fixed-v0-failures.txt`: the first fixed version failed two questions, found by the harness.
- `evidence/run3-set3-before-changes.txt`: before the answerability gate, the fixed system refused only 3 of 10 on-topic unanswerable questions in set 3. The earlier README figure of "1.00 correct abstention" was true only for off-topic questions.
- `evidence/run5-coverage-sweep.txt`: the gate's precision/recall trade-off. Strict thresholds refuse more traps but also refuse more valid paraphrases (at 0.6, 16/26 answerable pooled; at 0.5, 23/26).
- `evidence/mutation-run.txt`: 19 of 19 planted defects detected.
- `evidence/doctor-run.txt`: 7 findings on the Colab-style fixture (all six check codes), 0 on the Spaces-ready fixture.

## What this does NOT prove
- The sets are handwritten and tiny. The intervals are wide. These numbers are a demonstration of method, not a benchmark and not a population estimate.
- The gate is lexical. Known failures remain: it still answers some on-topic traps (for example "What is the delivery cost for orders under 50 dollars with standard delivery?") and refuses some valid paraphrases that use words absent from the FAQ ("shipping", "helpline", "allergy"). A real system needs semantic retrieval plus an answerability or verification step and a much larger evaluation set.
- The answer grader is crude: a phrase match with a three-word negation guard.
- Retrieval is TF-IDF over eight paragraphs: nothing here speaks to embeddings, large corpora, latency or cost.
- The doctor is static and regex-based: it cannot see runtime errors, so a clean report does not prove a Space will start. No deployment was performed or verified.
- No LLM and no real users.
