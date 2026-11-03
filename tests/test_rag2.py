"""Tests for rag2.RAGv2 (the real-data fix). One or more tests per rule, so a planted defect in any rule is caught.
Corpora here are tiny and invented for the test; no SQuAD question is used."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
import rag2  # noqa: E402
from rag2 import RAGv2, FEATURES, stem, tokens2, question_type, type_evidence, question_entities  # noqa: E402
from rag import load_docs  # noqa: E402
from evaluate import TUNED, HELDOUT, load_set3, load_set4, run  # noqa: E402

# Gate that answers whenever there is any candidate / never answers; used to look at retrieval alone.
OPEN = {"w": [10.0] + [0.0] * (len(FEATURES) - 1), "mean": [0.0] * len(FEATURES), "std": [1.0] * len(FEATURES), "threshold": 0.5}
SHUT = {**OPEN, "w": [-10.0] + [0.0] * (len(FEATURES) - 1)}

DOCS = [
    ("Harbor_p0", "The harbor of Valmora opened in 1850. Ships from the north docked at the long pier. "
                  "The harbor master was Thomas Greene."),
    ("Bakery_p0", "A bakery on Mill Street sells rye bread. The bakery opens at six. Bread costs 3 coins."),
    ("Clock_p0", "The clock tower has four faces. The clock tower was repaired in 1902."),
]


def feats(system, question):
    return system.candidate(question)[2]


def F(system, question, name):
    return feats(system, question)[FEATURES.index(name)]


class Text(unittest.TestCase):
    def test_stemming_lite(self):
        self.assertEqual(stem("countries"), "country")
        self.assertEqual(stem("walking"), "walk")
        self.assertEqual(stem("walked"), "walk")
        self.assertEqual(stem("boxes"), "box")
        self.assertEqual(stem("rivers"), "river")
        self.assertEqual(stem("class"), "class")      # double s kept
        self.assertEqual(stem("campus"), "campus")    # us kept
        self.assertEqual(stem("ring"), "ring")        # short words untouched
        self.assertEqual(stem("red"), "red")

    def test_tokens_drop_stopwords_and_possessives(self):
        self.assertEqual(tokens2("What was the king's castle?"), ["king", "castle"])
        self.assertEqual(tokens2("The Rivers of 1850"), ["river", "1850"])

    def test_question_types(self):
        self.assertEqual(question_type("How many ships docked?"), "num")
        self.assertEqual(question_type("What percentage was lost?"), "num")
        self.assertEqual(question_type("When was it built?"), "date")
        self.assertEqual(question_type("What year did it open?"), "date")
        self.assertEqual(question_type("Who built the pier?"), "proper")
        self.assertEqual(question_type("Where is the port?"), "proper")
        self.assertIsNone(question_type("What is a pier?"))
        self.assertIsNone(question_type("Why is it long?"))

    def test_type_evidence_num_date_proper(self):
        self.assertTrue(type_evidence("num", "It cost 3 coins.", "How much?"))
        self.assertTrue(type_evidence("num", "It had twelve masts.", "How many masts?"))
        self.assertFalse(type_evidence("num", "It had masts.", "How many masts?"))
        self.assertTrue(type_evidence("date", "It opened in 1850.", "When?"))
        self.assertTrue(type_evidence("date", "It opened in March.", "When?"))
        self.assertTrue(type_evidence("date", "In the 19th century it grew.", "When?"))
        self.assertFalse(type_evidence("date", "It opened early.", "When?"))
        self.assertFalse(type_evidence("date", "It had 3 masts.", "When?"))  # a short number is not a date
        self.assertTrue(type_evidence("proper", "The pier was built by Thomas Greene.", "Who built the pier?"))
        self.assertFalse(type_evidence("proper", "The pier was built long ago.", "Who built the pier?"))
        self.assertTrue(type_evidence(None, "anything", "What is it?"))

    def test_proper_name_must_not_be_a_question_word_or_sentence_start(self):
        # sentence-initial capital does not count; a capitalised word that the question itself uses does not count
        self.assertFalse(type_evidence("proper", "Ships docked at the pier.", "Who built the pier?"))
        self.assertFalse(type_evidence("proper", "The pier of Valmora was long.", "Where is Valmora?"))

    def test_question_entities_skip_first_word(self):
        self.assertEqual(question_entities("Valmora has a harbor in Europe?"), ["europe"])
        self.assertEqual(question_entities("Where is Valmora harbor?"), ["valmora"])
        self.assertEqual(question_entities("what is the pier"), [])


class Retrieval(unittest.TestCase):
    def setUp(self):
        self.s = RAGv2(DOCS, gate=OPEN)

    def test_best_sentence_is_returned_alone(self):
        r = self.s.answer("Who was the harbor master?")
        self.assertFalse(r["abstained"])
        self.assertEqual(r["answer"], "The harbor master was Thomas Greene.")
        self.assertEqual(r["citations"], ["Harbor_p0"])

    def test_rare_word_outweighs_common_word(self):
        # "bakery" appears in two sentences, "rye" in one: the rare word decides
        r = self.s.answer("rye bakery")
        self.assertIn("rye", r["answer"])

    def test_paragraph_context_lifts_a_sentence(self):
        # the sentence has no match for "Valmora" itself; the paragraph does
        with_ctx = RAGv2(DOCS, retrieval={"w_par": 1.0, "w_type": 0.0}, gate=OPEN)
        no_ctx = RAGv2(DOCS, retrieval={"w_par": 0.0, "w_type": 0.0}, gate=OPEN)
        q = "Valmora ships pier"
        c1 = with_ctx.rank(q)[0][0][0]
        c0 = no_ctx.rank(q)[0][0][0]
        self.assertGreater(c1, c0)

    def test_title_words_are_indexed(self):
        s = RAGv2([("Other_p0", "It lives on the plain."), ("Zebra_p0", "It lives on the plain.")], gate=OPEN)
        r = s.answer("zebra plain")
        self.assertEqual(r["citations"], ["Zebra_p0"])

    def test_type_bonus_prefers_sentence_with_the_needed_type(self):
        docs = [("A_p0", "The tower was repaired in 1902. The tower was repaired by workers.")]
        s = RAGv2(docs, retrieval={"w_type": 0.5}, gate=OPEN)
        self.assertIn("1902", s.answer("When was the tower repaired?")["answer"])
        s2 = RAGv2([("A_p0", "The tower was repaired by workers. The tower was repaired in 1902.")], retrieval={"w_type": 0.5}, gate=OPEN)
        self.assertIn("1902", s2.answer("When was the tower repaired?")["answer"])
        # without the bonus the first sentence wins the tie
        s3 = RAGv2([("A_p0", "The tower was repaired by workers. The tower was repaired in 1902.")], retrieval={"w_type": 0.0}, gate=OPEN)
        self.assertNotIn("1902", s3.answer("When was the tower repaired?")["answer"])

    def test_equal_scores_are_broken_by_sentence_order(self):
        s = RAGv2([("D_p0", "Apple pear. Plum fig.")], gate=OPEN)
        self.assertEqual(s.answer("plum apple")["answer"], "Apple pear.")

    def test_no_matching_words_abstains(self):
        r = self.s.answer("quantum chromodynamics")
        self.assertTrue(r["abstained"])
        self.assertEqual(r["citations"], [])

    def test_question_without_content_words_abstains(self):
        self.assertTrue(self.s.answer("What is the?")["abstained"])

    def test_empty_corpus(self):
        self.assertTrue(RAGv2([], gate=OPEN).answer("harbor")["abstained"])


class Features(unittest.TestCase):
    def setUp(self):
        self.s = RAGv2(DOCS, gate=OPEN)

    def test_vector_matches_feature_names(self):
        self.assertEqual(len(feats(self.s, "Who was the harbor master?")), len(FEATURES))

    def test_coverage_full_and_partial(self):
        self.assertAlmostEqual(F(self.s, "harbor master Greene", "cov_sent"), 1.0)
        self.assertLess(F(self.s, "harbor master Greene submarine", "cov_sent"), 1.0)

    def test_unseen_rare_word_lowers_coverage_and_raises_rare_miss(self):
        a = "harbor master Greene"
        b = "harbor master Greene submarine"
        self.assertGreater(F(self.s, b, "rare_miss"), F(self.s, a, "rare_miss"))
        self.assertGreater(F(self.s, b, "miss_par_cnt"), F(self.s, a, "miss_par_cnt"))
        self.assertLess(F(self.s, b, "cov_par"), F(self.s, a, "cov_par"))

    def test_margin_zero_when_two_sentences_tie_and_positive_when_clear(self):
        tie = RAGv2([("A_p0", "Red apples grow. Red apples grow.")], gate=OPEN)
        self.assertAlmostEqual(F(tie, "red apples", "margin"), 0.0)
        self.assertGreater(F(self.s, "harbor master Greene", "margin"), 0.1)

    def test_entity_missing_from_paragraph(self):
        self.assertAlmostEqual(F(self.s, "Did the harbor master meet Zorblax", "ent_miss"), 1.0)
        self.assertAlmostEqual(F(self.s, "Did the harbor master meet Greene", "ent_miss"), 0.0)
        self.assertAlmostEqual(F(self.s, "Did the Harbor master meet Greene", "ent_miss_sent"), 0.0)
        self.assertGreater(F(self.s, "Is the Harbor master in Bakery", "ent_miss_sent"), 0.0)

    def test_number_missing_from_paragraph(self):
        self.assertAlmostEqual(F(self.s, "harbor opened 1850", "num_miss"), 0.0)
        self.assertAlmostEqual(F(self.s, "harbor opened 1999", "num_miss"), 1.0)

    def test_type_miss(self):
        self.assertEqual(F(self.s, "When did the bakery open", "type_miss"), 1.0)
        self.assertEqual(F(self.s, "When was the harbor opened", "type_miss"), 0.0)
        self.assertEqual(F(self.s, "What does the bakery sell", "type_miss"), 0.0)

    def test_bigram_coverage_needs_adjacent_words(self):
        adj = F(self.s, "clock tower faces", "bigram_sent")
        swapped = F(self.s, "faces tower clock", "bigram_sent")
        self.assertGreater(adj, swapped)
        self.assertGreater(F(self.s, "clock tower faces", "bigram_par"), F(self.s, "faces tower clock", "bigram_par"))

    def test_compactness(self):
        s = RAGv2([("A_p0", "Red apples and green pears grew. Red x y z w green grew.")], gate=OPEN)
        tight = F(s, "red apples", "compact")
        wide = F(s, "red green", "compact")
        self.assertAlmostEqual(tight, 1.0)
        self.assertLess(wide, 1.0)

    def test_matched_and_missing_rare_word_features(self):
        self.assertGreater(F(self.s, "harbor master Greene", "matched_rare"), 0.0)
        self.assertGreater(F(self.s, "harbor submarine", "miss_sent_rare"), F(self.s, "harbor master", "miss_sent_rare"))
        self.assertGreater(F(self.s, "harbor master Greene", "n_terms"), F(self.s, "harbor", "n_terms"))

    def test_squared_coverage_feature(self):
        v = feats(self.s, "harbor master Greene submarine")
        self.assertAlmostEqual(v[FEATURES.index("cov_sent_sq")], v[FEATURES.index("cov_sent")] ** 2)


class Gate(unittest.TestCase):
    def test_threshold_decides(self):
        q = "Who was the harbor master?"
        self.assertFalse(RAGv2(DOCS, gate=OPEN).answer(q)["abstained"])
        self.assertTrue(RAGv2(DOCS, gate=SHUT).answer(q)["abstained"])

    def test_abstained_answer_still_exposes_the_candidate(self):
        r = RAGv2(DOCS, gate=SHUT).answer("Who was the harbor master?")
        self.assertIn("Thomas Greene", r["candidate"])
        self.assertEqual(r["candidate_doc"], "Harbor_p0")
        self.assertEqual(r["citations"], [])

    def test_probability_is_logistic_in_standardised_features(self):
        s = RAGv2(DOCS, gate={"w": [0.0, 1.0] + [0.0] * (len(FEATURES) - 2), "mean": [0.0, 0.5] + [0.0] * (len(FEATURES) - 2),
                              "std": [1.0, 0.25] + [1.0] * (len(FEATURES) - 2), "threshold": 0.5})
        x = [1.0, 0.75] + [0.0] * (len(FEATURES) - 2)
        self.assertAlmostEqual(s.probability(x), 1 / (1 + 2.718281828459045 ** -1.0), places=6)
        x[1] = 0.25
        self.assertLess(s.probability(x), 0.5)

    def test_threshold_is_a_dial(self):
        q = "Who was the harbor master?"
        base = {**OPEN, "w": [1.0] + [0.0] * (len(FEATURES) - 1)}   # probability is 0.731 for every question
        self.assertTrue(RAGv2(DOCS, gate={**base, "threshold": 0.75}).answer(q)["abstained"])
        self.assertFalse(RAGv2(DOCS, gate={**base, "threshold": 0.70}).answer(q)["abstained"])

    def test_frozen_gate_is_consistent(self):
        g = rag2.FROZEN_GATE
        self.assertEqual({len(g["w"]), len(g["mean"]), len(g["std"])}, {len(FEATURES)})
        self.assertTrue(all(s > 0 for s in g["std"]))
        self.assertTrue(0 < g["threshold"] < 1)

    def test_default_gate_is_the_frozen_one(self):
        self.assertIs(RAGv2(DOCS).gate, rag2.FROZEN_GATE)


class FaqRegression(unittest.TestCase):
    """The frozen v2 on the old grocer FAQ. These numbers are what executed code printed (evidence/v2-04-faq-eval.txt);
    they are pinned so a change is visible. The v2 gate was fitted on Wikipedia text, not on this FAQ."""

    @classmethod
    def setUpClass(cls):
        from rag import load_docs as ld
        with open(os.path.join(ROOT, "data", "faq.txt"), encoding="utf-8") as f:
            cls.docs = ld(f.read())

    def test_pinned_counts(self):
        s = RAGv2(self.docs)
        got = {name: (run(s, g)["answerable_correct"], run(s, g)["unanswerable_abstained"])
               for name, g in (("tuned", TUNED), ("heldout", HELDOUT), ("set3", load_set3()), ("set4", load_set4()))}
        self.assertEqual(got, FAQ_PINNED)


FAQ_PINNED = {"tuned": ((3, 10), (4, 4)), "heldout": ((1, 6), (2, 2)), "set3": ((1, 10), (10, 10)), "set4": ((3, 8), (9, 9))}
# (answerable correct k, n), (unanswerable abstained k, n). The old RAG scores 10/10, 6/6, 7/10, 8/8 answerable on the same sets.


if __name__ == "__main__":
    unittest.main()
