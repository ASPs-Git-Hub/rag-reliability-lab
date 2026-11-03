# Spot check, by hand (one reader, me, 2026-11-03)

Sample: 30 unanswerable questions that fixed RAG answered, drawn with `random.Random(20261103).sample` (see the end of `01-frozen-run.txt`, items 1 to 30).
Question I asked for each: does the returned text actually answer the question as asked?

- Looks answerable (5): 1 (Palm Springs popular for resort feel), 16 (X.25 simpler host interface), 17 (about 3,000 slaughtered in Toulouse), 26 (heat comes most commonly from burning fuel), 28 (12 California State University campuses).
- Borderline (2): 11 (words swapped: "highly concentrated sources of oxygen promote rapid combustion"), 29 (garbled question, text lists examples of function problems).
- Not answerable from the returned text (23): the rest. Typical reason: the question changes one detail (a date, a number, a place) that the text does not have.

Count: 5 of 30 clearly answerable (Wilson 95% CI 7%-34%); 7 of 30 if borderline cases are counted (12%-41%). A single reader judged; no second opinion.
