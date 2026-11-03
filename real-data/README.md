# RAG Reliability Lab: run on real data

**Independent technical demonstration on a public research dataset. Not a client result.**

Status: BUILT_UNVERIFIED. I ran it once, myself, on 2026-11-03. Nobody else has reproduced it.

## What this is
The main lab tests a small question-answering program on a made-up grocer FAQ. Made-up data is easy to pass. So I ran the same program, with no changes, on real data.

Words I use:
- **RAG** means the program looks up text from a pile of documents and answers from it.
- **Abstain** means the program says "I don't know" instead of guessing.
- **Wilson 95% interval** is a range. If I did the test again with other questions, the true score would probably land inside it.

## The data
- **SQuAD 2.0**, the development set, from Stanford. Link and file hash are in `SOURCES.md`.
- Licence: CC BY-SA 4.0. The texts are Wikipedia paragraphs.
- It has two kinds of questions. Some have an answer in the text. Others were written by people to look like they have an answer, but the text does not say it. The program should refuse those.
- I took the first 20 articles in the file. That is 621 paragraphs, 3,068 questions with answers and 3,179 questions without. All went into one pile, so the program had to find the right paragraph on its own.

## What I ran
1. I wrote the rules first, in `PROTOCOL.md`. I did not change them after seeing results.
2. `run_eval.py` loads the data and asks every question to two programs from the lab: the fixed one (`RAG`, default settings) and the deliberately weak one (`BrokenRAG`).
3. The fixed program was scored with its settings exactly as shipped. I did not tune anything before this run.
4. A question with an answer counts as right only if the program answered and the text it returned contains a correct answer string. A question without an answer counts as right only if the program refused.
5. Raw output is in `evidence/01-frozen-run.txt`.

## Results (the frozen run)

| | Fixed RAG | BrokenRAG |
|---|---|---|
| Questions with an answer: answered correctly | 1300 of 3068 = 42.4% (95% interval 40.6% to 44.1%) | 585 of 3068 = 19.1% (17.7% to 20.5%) |
| Same, looser rule (the cited paragraph holds the answer) | 1409 of 3068 = 45.9% (44.2% to 47.7%) | not applicable, it cites nothing |
| Questions with an answer: refused | 1449 = 47.2% | 0 |
| Questions without an answer: refused | 1879 of 3179 = 59.1% (57.4% to 60.8%) | 0 of 3179 = 0% (0.0% to 0.1%) |

Plain reading: the fixed program is clearly better than the broken one. It is also wrong a lot. It refuses almost half of the questions that have an answer. And it answers about 4 in 10 of the questions that it should refuse.

## What it gets wrong
Counts for the fixed program, with real examples (Wikipedia text, short quotes, article named in each id):

| Kind of mistake | Count | Example |
|---|---|---|
| Refused a question that has an answer, because the best match scored too low | 1256 | "In what country is Normandy located?" (answer: France). Match score 0.25, limit 0.30. |
| Refused, because too few question words were found in the paragraph | 193 | "What river was Petrela located by?" (answer: Deabolis). |
| Answered from the wrong paragraph | 210 | "What was the Norman religion?" returned a sentence about the French Wars of Religion destroying Norman art (Normans_p34). |
| Right paragraph, wrong sentence picked | 109 | "Who made Ralph earl?" returned a sentence about the 1051 appointments. The answer, Edward the Confessor, was in the paragraph but not in the sentences it chose. |
| Should refuse, but answered from the paragraph the question was written about | 1100 | "Who did King Charles III swear fealty to?" The text says Rollo swore fealty to King Charles III, so the words match, and the answer is the wrong way round. |
| Should refuse, but answered from another paragraph | 200 | "What fell to the Normans in the 10th century?" returned a sentence about the Normans' identity emerging in the 10th century (Normans_p0). |

The biggest cause is plain. The program judges by shared words. Real questions often use different words from the text ("country" vs "France"), so good matches score low. The made-up questions in SQuAD 2.0 reuse the text's own words with one detail changed (a date, a name, a swap), so bad matches score high. A word-overlap test cannot tell these apart. Five examples of each kind are in the evidence file.

## Spot check: are the "unanswerable" ones really unanswerable?
With 20 articles in one pile, some questions marked unanswerable might be answerable from another paragraph. I drew 30 at random from the 1300 that the fixed program answered, and read them (`evidence/03-spot-check-by-hand.md`). 5 of 30 are clearly answerable from the returned text (95% interval 7% to 34%) and 2 more are borderline. So the 59.1% refusal score is slightly too low, but most of the answered ones are real misses. One reader only.
I did not check the other direction (questions with answers that were refused).

## After seeing results: a settings sweep
This part is tuned on the same questions, so it is optimistic. It is not the headline. Raw output: `evidence/02-after-seeing-sweep.txt`.

| Match-score limit | Answers found | Unanswerable refused |
|---|---|---|
| 0.1 | 60.0% | 25.1% |
| 0.2 | 57.0% | 33.6% |
| 0.3 (default) | 42.4% | 59.1% |
| 0.4 | 22.7% | 82.1% |
| 0.5 | 7.5% | 94.5% |
| 0.6 | 1.6% | 98.6% |

Changing the other setting (share of question words found, 0.0 to 1.0) gives the same shape: from 44.3% found and 55.1% refused at 0.0, to 10.8% found and 92.8% refused at 1.0. No setting does well on both. At 0.1 the program finds 60% of answers and refuses 25% of the unanswerable ones. Raising the limit buys refusals only by losing answers.

## What this does NOT prove
- It does not show that a better RAG (with embeddings or a language model) fails or works here. This program only counts shared words.
- The scoring is a text match. A correct answer worded differently from the gold answer counts as wrong, and a wrong answer that happens to contain the gold string counts as right.
- 20 Wikipedia articles are not a business's documents. The result says nothing about a client's data.
- The 59.1% is for questions made to fool programs. Real customers' unanswerable questions are probably easier, but I have no data on that.
- The spot check is 30 items read by one person.
- The sweep was tuned on the test questions.
- The lab's original tests (27 tests, 19 planted defects) are unchanged. They were rerun together with the v2 tests (see the v2 section below).
- The text "I could not find that in the FAQ." is built into the lab and shows in this run's refusals. I did not change it.

## Deviations
None. The protocol was followed as written. The run took about 2 minutes (fixed, 32 s; broken, 86 s) on Python 3.14.

## v2: what I changed and what happened
**Status: BUILT_UNVERIFIED.** I ran it once on 2026-11-03, myself. Nobody else has reproduced it. Rules written first: `PROTOCOL-v2.md`.

### The problem, in plain words
The first program (v1) counted how many of the question's words appeared in a paragraph. Real questions often use other words than the text ("country" against "France"), so good matches looked weak. The trick questions in SQuAD reuse the text's words with one detail changed, so bad matches looked strong. Counting shared words cannot tell the two apart.

### What I changed
Everything is in `src/rag2.py` (class `RAGv2`). The old `RAG` and `BrokenRAG` are untouched.
1. **Better search.** v1 compared whole paragraphs. v2 scores each sentence with **BM25** (a standard formula: a rare shared word counts more than a common one, and long sentences are not favoured), then adds a share of the paragraph's score, so a sentence can borrow the topic from its paragraph. Words are cut back to a root ("countries" and "country" meet).
2. **Question type.** "When" wants a year or month, "how many" wants a number, "who" and "where" want a capitalised name. A sentence that has the right kind of thing gets a small bonus.
3. **One sentence back.** v2 returns the best sentence only. v1 sometimes returned two.
4. **A new "should I answer?" rule** (the **gate**). It looks at 17 signs of whether the best sentence really backs the question. Examples: how much of the question's rare words it covers, how many neighbouring word pairs match, whether a name or number from the question is missing from the paragraph, whether the sentence has the type of thing asked for, and how far it beats the second-best sentence. A **logistic regression** (a method that turns such signs into a probability) combines them. The program refuses when the probability is under 0.50.
5. **Fitted on different articles.** Search settings, the weights and the 0.50 limit were all chosen on articles 21 to 35 of the same file. The test articles (1 to 20) were not used for any choice, example or default. I ran the test once.

### Result (same 3,068 answerable and 3,179 unanswerable questions as v1; raw: `evidence/v2-03-frozen-run.txt`)

| | v1 before | v2 after |
|---|---|---|
| Answerable, answered correctly | 1300 = 42.4% (95% interval 40.6% to 44.1%) | 1029 = 33.5% (31.9% to 35.2%) |
| Unanswerable, refused | 1879 = 59.1% (57.4% to 60.8%) | 2384 = 75.0% (73.5% to 76.5%) |
| Mean of the two rates | 50.7% | 54.3% |

Plain reading: v2 refuses many more trick questions (about 16 points more) and gives up about 9 points of right answers to get that. The mean of the two rises 3.6 points, a modest gain, not a fix. Whether that trade is good depends on what a wrong answer costs. The settings I froze lean toward refusing.

The limit can be moved. The curve below is the same frozen scorer at other limits, read on the test questions without re-choosing anything (descriptive only):

| Refuse below | Answerable correct | Unanswerable refused |
|---|---|---|
| 0.30 | 62.5% | 38.3% |
| 0.40 | 49.1% | 60.2% |
| 0.45 | 41.3% | 68.7% |
| 0.50 (frozen) | 33.5% | 75.0% |
| 0.60 | 21.0% | 85.0% |
| 0.70 | 9.9% | 93.7% |

At 0.40, v2 refuses 60.2% (v1: 59.1%) and answers 49.1% correctly (v1: 42.4%), so at the same refusal level it finds about 7 points more. At v1's answer rate (42.4%) it refuses 60.2%, about 1 point more than v1. So the gain is real at the refusing end and almost nothing at the answering end.

Search alone is better: if v2 never refused, its top sentence holds the answer for 71.3% of answerable questions (69.7% to 72.9%). The loss comes from the gate.

### Kinds of mistake, before and after (same rules as the v1 table)

| Kind of mistake | v1 | v2 |
|---|---|---|
| Answerable, refused | 1449 | 1932 (of these, 1158 had the right sentence and 774 did not) |
| Answerable, answered from the wrong paragraph | 210 | 51 |
| Answerable, right paragraph, wrong sentence | 109 | 56 |
| Unanswerable, answered from its own paragraph | 1100 | 743 |
| Unanswerable, answered from another paragraph | 200 | 52 |

Paired by question: of the answerable ones, 733 were right in both, 567 only in v1 and 296 only in v2. Of the unanswerable ones, 759 were refused only by v2 and 254 only by v1.

### What still fails
- 743 trick questions (23%) still get an answer. They reuse the paragraph's words, so a word-based gate has nothing to catch. The example in the v1 section (who swore fealty to whom) is still out of reach.
- 1158 questions with an answer are refused although v2 had found the right sentence. The gate cannot tell them from trick questions with the signs it has.
- 28.7% of answerable questions never get the right sentence even when v2 is forced to answer.
- This is still counting words. A model that reads meaning would be the next step; I did not add one.

### Old grocer FAQ (`evidence/v2-04-faq-eval.txt`)
The v2 gate was fitted on Wikipedia text and not on this FAQ, and it got worse there. Answerable correct, v1 against v2: tuned set 10/10 to 3/10, second set 6/6 to 1/6, frozen set 3 7/10 to 1/10, frozen set 4 8/8 to 3/8. Unanswerable refused: 4/4, 2/2 and 4/4 stay the same on the first two sets; set 3 goes from 7/10 to 10/10 and set 4 from 6/9 to 9/9. So on the FAQ v2 refuses almost everything. FAQ questions are short and a gate fitted on long Wikipedia questions reads them as weakly supported. A gate fitted per knowledge base would be needed. The old `RAG` is unchanged and still gets its old numbers.

### Quality checks
- Tests: 60 pass (the 27 original, unchanged, plus 33 new in `tests/test_rag2.py`). I changed no original test. Raw: `evidence/v2-05-unittest.txt`.
- Planted defects (`scripts/mutate.py`): 41 of 41 caught (the 19 old ones plus 22 for v2). First attempt caught 39 of 43. Four survived: two were harmless code (a text replace and an input guard that did nothing, so I removed both lines) and two needed a better test (title words, and the order of equal scores). Raw: `evidence/v2-06-mutation.txt`.
- Tuning evidence: `evidence/v2-01-tuning.txt`, `evidence/v2-01b-feature-cv.txt`, `evidence/v2-02-tuning-dry-run.txt`. On the tuning articles v2 scored 37.7% and 78.5% (optimistic, fitted there); v1 on the same articles scored 45.1% and 59.6%.

### Deviations and notes
- The file has 35 articles, not 40, so tuning used articles 21 to 35 (15 articles). Stated in `PROTOCOL-v2.md` before the test run.
- The protocol says v1's mean is 50.8%. From unrounded counts it is 50.7%. The reading does not change.
- The test run used the exact file versions listed in `PROTOCOL-v2.md` (hashes printed at the top of the evidence file). Its v1 numbers reproduce the first run exactly (1300 and 1879).
- The 30-question hand check was not repeated for v2. The sample is printed at the end of `evidence/v2-03-frozen-run.txt` and has not been read.
- `src/evaluate.py` got a `--v2` flag for the FAQ run; its other behaviour is unchanged.
- The scoring is still a text match on the returned sentence, which favours a system that returns more text. v2 returns less text than v1, so this works against v2 if anything.
- Reproduce: `./fetch_data.sh <new-empty-dir>`, then `python3 -I run_eval_v2.py <dir>/dev-v2.0.json --split test` (about 2 minutes). Refit on the tuning articles: `--split tune --out params.json`.

## Files (v1)
`PROTOCOL.md`, `SOURCES.md`, `SHA256.txt`, `fetch_data.sh`, `run_eval.py`, `frozen_question_ids.txt` (all 6,247 question ids, flag A or U), `evidence/`.
Reproduce: `./fetch_data.sh <new-empty-dir>`, then `python3 -I run_eval.py <dir>/dev-v2.0.json` (add `--sweep` for the sweep).
