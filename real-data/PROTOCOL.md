# Protocol, written before the first scored run (2026-11-03)

Nothing below is changed after results are seen. If a deviation happens, it goes in README.md under "Deviations", with the reason.

## Data
SQuAD 2.0 dev set, file `dev-v2.0.json` (version "v2.0", 35 articles), sha256 in `SHA256.txt`. Licence CC BY-SA 4.0 (stated on the SQuAD explorer page).

## Corpus (one knowledge base)
- Rule: the first 20 articles in file order.
- Every paragraph of those articles is one document. Document id = article title with every non-word character replaced by `_`, plus `_p<paragraph index>` (for example `Normans_p0`). The lab's title boost reads the id, so the article title acts as a title.
- All paragraphs go into ONE index. Questions are not told which article they belong to.

## Questions
- All questions of those 20 articles, answerable and unanswerable (`is_impossible`), in file order. No sampling.
- Frozen id list: `frozen_question_ids.txt` (one id per line, with a flag A or U).

## Systems under test (frozen)
- `RAG(docs)` with the lab's UNCHANGED defaults: threshold 0.30, window 3, min_coverage 0.5.
- `BrokenRAG(docs)` with its defaults (width 70).
- `src/` is imported as it is. Nothing in `src/` is edited.

## Scoring (decided now)
Text normalisation (SQuAD standard): lowercase, remove punctuation, remove the words a / an / the, collapse spaces.
- Answerable question is CORRECT only if the system did not refuse AND the normalised returned answer text contains, at word boundaries, at least one normalised gold answer string. (Headline rule. "Returned passage" means the sentences the system returns.)
- Secondary, looser rule, reported next to it and never as the headline: not refused AND the cited paragraph (whole paragraph) contains a gold answer string. BrokenRAG gives no citation, so for BrokenRAG only the headline rule applies (its returned text is the 70-character chunk).
- Unanswerable question is CORRECT only if the system refuses.
- Counts are reported with the lab's own `wilson()` 95% interval, for both systems, for answerable and unanswerable separately.

## Caveats stated in advance
- In a 20-article corpus a few "unanswerable" questions may be answerable from another paragraph. Check: 30 unanswerable questions that RAG answered (not refused), drawn with `random.Random(20261103).sample`, read by hand. For each, I judge whether the returned text answers the question. I report how many look genuinely answerable.
- The unanswerable questions in SQuAD 2.0 were written by crowd workers to look like the answerable ones about the same paragraph. They are hard on purpose for a word-overlap gate.

## Failure categories (rules fixed now, applied by code)
Answerable, RAG:
1. `refused_low_score`: refused and top-chunk score < threshold.
2. `refused_low_coverage`: refused and score >= threshold but coverage < min_coverage.
3. `wrong_paragraph`: answered, cited paragraph is not one that contains a gold answer string.
4. `right_paragraph_wrong_sentence`: answered, cited paragraph contains a gold string, returned sentences do not.
Unanswerable, RAG:
5. `answered_from_own_paragraph`: answered, cited paragraph is the paragraph the question was written about.
6. `answered_from_other_paragraph`: answered, cited paragraph is another one.
Five short real examples per category are printed by the script (first five in file order).

## After-seeing-results part (clearly labelled, not the headline)
A sweep on this same question set, reported as a trade-off table: min_coverage in {0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0} at threshold 0.30, and threshold in {0.10, 0.20, 0.30, 0.40, 0.50, 0.60} at min_coverage 0.5. These numbers are tuned on the test questions, so they are optimistic and are not a result.
