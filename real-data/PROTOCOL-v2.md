# Protocol v2, written before the frozen v2 run (2026-11-03)

Goal: fix the failure found in the first real-data run (`PROTOCOL.md`, `evidence/01-frozen-run.txt`: 42.4% of answerable questions answered correctly, 59.1% of unanswerable questions refused) and measure the fix on the same frozen questions. Success is not promised. A small gain, no gain, or a loss are all reportable results and will be reported as they come out.

## What stays fixed
- The "before" is the original frozen v1 result (`RAG` with its shipped defaults, `evidence/01-frozen-run.txt`). It is not re-tuned. The v2 run recomputes v1 on the same questions as a check that nothing drifted (expected: 1300 and 1879).
- `src/rag.py` (including `BrokenRAG`) is untouched.
- Test set: the 20 first articles of SQuAD 2.0 dev, the same 6,247 questions as `frozen_question_ids.txt` (sha256 `dbdbf6db4a352444d0232a950ec0645cfe474a7d9e4217da7e5695add53e6c22`), one index of 621 paragraphs.
- Scoring rules and normalisation: exactly as in `PROTOCOL.md`.

## What is tuned, and where
Tuning set: SQuAD 2.0 dev, every article after the first 20, same file, same fetch script. **Deviation from the plan, stated now:** the file has 35 articles, not 40, so the tuning set is articles 21 to 35: 15 articles, 583 paragraphs, 2,860 answerable and 2,766 unanswerable questions, one index. It has no article or paragraph in common with the test set. Its index is a little smaller than the test index (583 against 621 paragraphs).

Tuned ONLY on the tuning set (`evidence/v2-01-tuning.txt`):
1. Retrieval settings of `RAGv2`, chosen by a grid (BM25 k1 in {0.9, 1.2, 1.8}, b in {0.5, 0.75}, paragraph-context weight in {0, 0.3, 0.6, 1.0}, answer-type bonus in {0, 0.1, 0.2}). Criterion: the share of answerable tuning questions whose top sentence contains a gold answer when the system is forced to answer. Chosen: k1 0.9, b 0.5, paragraph weight 0.6, type bonus 0.1.
2. The abstain rule: a logistic regression over 17 features plus a bias (`FEATURES` in `src/rag2.py`), fitted by gradient descent. Training rows: answerable questions whose top sentence is right (label answer, weight 1/answerable) and all unanswerable questions (label refuse, weight 1/unanswerable). Answerable questions whose top sentence is wrong are left out because answering and refusing score the same for them. The weights are standardised and frozen in `src/rag2.py` (`FROZEN_GATE`).
3. The refusal threshold on the gate probability, chosen on the tuning set from a grid 0.05 to 0.95 in steps of 0.05 to maximise the mean of the two rates (below). Chosen: 0.50.

Also done on the tuning set only, and disclosed because it shaped the final feature list: the feature set grew from 10 to 18 (incl. bias) after a 2-fold cross-validation by article on the tuning set (`evidence/v2-01b-feature-cv.txt`, scripts in `tuning/`). A boosted-stumps model was tried in the same cross-validation and did not beat the logistic regression, so it was not used. Six hand-written rules were set before looking at any data: question-type patterns, the number words and month list, the stop-word list, the stemming rules, the capitalised-name rule, and one returned sentence (not two).

Not touched by any tuning: the test articles. No test question was used in any default, example, test or fit. The unit tests use invented text only.

## The exact metric (declared now)
- Answerable correct: not refused AND the returned sentence contains a normalised gold answer string at word boundaries (headline, same as v1). Returned text for v2 is one sentence.
- Unanswerable refused: the system refused.
- Objective used to choose the threshold: the mean of the two rates, (answerable correct rate + unanswerable refused rate) / 2, because the two failures cost about the same for a support bot and neither alone is the goal. v1's value of this mean on the test set is (42.4% + 59.1%) / 2 = 50.8%.
- Reported with Wilson 95% intervals for each of the two rates, for v1 and v2, on the same 3,068 and 3,179 questions.
- Also reported: error categories before and after (same rules as `PROTOCOL.md`, plus a split of v2's refusals by whether its sentence was right), the paired counts per question, and the trade-off curve of the frozen v2 scorer across thresholds. The curve is descriptive only: the threshold is NOT re-chosen on test. Two comparisons are read off it at equal footing with v1 (best v2 answerable rate at v1's refusal rate or higher; best v2 refusal rate at v1's answerable rate or higher), labelled as read from a curve, not as a frozen result.

## Reading rule, fixed before the run
- The frozen v2 point is threshold 0.50 and nothing else.
- v2 counts as better on the objective only if its mean of the two rates exceeds v1's 50.8% and both of its rates are reported with their intervals. If the mean rises by under 3 points I call it a small gain. If it does not rise I say it did not help. I will not pick another threshold, feature set or retrieval setting after seeing the test result. If I change anything, it goes in README.md under "Deviations" and the changed system is not called frozen.
- A caveat I expect to apply: the frozen point trades answerable questions for refusals (the tuning curve shows this). A reader who values finding answers more than refusals should read the curve instead of the single point.

## The run
- Command: `python3 -I run_eval_v2.py <dev-v2.0.json> --split test`, run once; raw output kept in `evidence/v2-03-frozen-run.txt`.
- Files that must be the ones used (full sha256): `src/rag.py` 5f4e935ba6039e06f5c8f5d067379c34d46ed01faddf74ff76202bc9d6978097; `src/rag2.py` 9e640c50933d5540814d2e2c6aa52fe13ef3eb553ff2691613bea0d320af45cb; `run_eval_v2.py` ab44e096c4bd0eaa0589592971d36cefe06ab38387891bff4478c7998aec5552; `report_v2.py` 0e086a77058e1a1daa34538d7f49c5ae6ed94a132f15572c73ecc861e884265b. The run prints the first 16 characters of each and of this file.
- Quality gates run before the test run, on invented data and on the tuning split only: the 27 original tests plus the new `tests/test_rag2.py` (33 tests), `scripts/mutate.py`, the grocer FAQ evaluation (`evidence/v2-04-faq-eval.txt`, the v2 gate was not fitted on the FAQ), and a dry run of the report on the tuning split (`evidence/v2-02-tuning-dry-run.txt`).

## Caveats stated in advance
- This is still a word-overlap system. It has no meaning model. It will not understand that a crowd-worker swapped one detail in a trick question. The expected ceiling is modest.
- The 30-question hand check of "unanswerable" questions that are in fact answerable (done for v1) is not repeated for v2 in this run; v2's sample is printed but not yet read.
- The tuning set is smaller than the plan (15 articles, not 20).
- Gold-string matching undercounts correct answers worded differently and overcounts lucky matches; unchanged from v1.
