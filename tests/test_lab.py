import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from rag import BrokenRAG, RAG, load_docs, norm_tokens  # noqa: E402
from evaluate import TUNED, HELDOUT, load_set3, load_set4, run, phrase_ok, wilson, fmt  # noqa: E402
from doctor import diagnose, check_requirements  # noqa: E402

with open(os.path.join(ROOT, "data", "faq.txt"), encoding="utf-8") as _f:
    DOCS = load_docs(_f.read())
POOLED = TUNED + HELDOUT + load_set3() + load_set4()


def codes(issues):
    return {c for c, _ in issues}


class Retrieval(unittest.TestCase):
    def test_docs_loaded(self):
        self.assertEqual(len(DOCS), 8)

    def test_normalisation_matches_punctuation_and_case(self):
        self.assertEqual(norm_tokens("Days."), norm_tokens("days"))

    def test_empty_corpus_abstains_instead_of_crashing(self):
        r = RAG([]).answer("hi")
        self.assertTrue(r["abstained"])

    def test_invalid_window_rejected_with_a_clear_message(self):
        for bad in (0, -1, 1.5, "2"):
            with self.assertRaisesRegex(ValueError, "window must be a positive integer"):
                RAG(DOCS, window=bad)


class Contracts(unittest.TestCase):
    """The broken system must fail each contract; the fixed system's measured numbers are pinned so any
    change to them is visible. Pinned numbers are what evidence/run6-final-evaluation.txt shows."""

    def setUp(self):
        self.fixed, self.broken = RAG(DOCS), BrokenRAG(DOCS)

    def test_fixed_pinned_on_tuned_set(self):
        m = run(self.fixed, TUNED)
        self.assertEqual(m["answerable_correct"], (10, 10))
        self.assertEqual(m["unanswerable_abstained"], (4, 4))
        self.assertEqual(m["citations_correct"], (10, 10))

    def test_fixed_pinned_on_clean_frozen_set4(self):
        m = run(self.fixed, load_set4())
        self.assertEqual(m["answerable_correct"], (8, 8))
        self.assertEqual(m["unanswerable_abstained"], (6, 9))  # known: 3 on-topic traps still answered
        self.assertEqual(m["citations_correct"], (8, 8))

    def test_broken_never_abstains_on_any_set(self):
        for golden in (TUNED, HELDOUT, load_set3(), load_set4()):
            self.assertEqual(run(self.broken, golden)["unanswerable_abstained"][0], 0)

    def test_broken_rarely_answers_correctly(self):
        k, n = run(self.broken, TUNED)["answerable_correct"]
        self.assertLessEqual(k / n, 0.2)

    def test_broken_never_cites(self):
        self.assertEqual(run(self.broken, POOLED)["citations_correct"][0], 0)

    def test_the_answerability_gate_is_what_abstains(self):
        """With the gate off, the fixed system abstains on clearly fewer on-topic traps."""
        on = run(RAG(DOCS), POOLED)["unanswerable_abstained"][0]
        off = run(RAG(DOCS, min_coverage=0.0), POOLED)["unanswerable_abstained"][0]
        self.assertGreater(on, off + 3)

    def test_return_address_trap_now_abstains(self):
        self.assertTrue(self.fixed.answer("What is your return address in Canada?")["abstained"])

    def test_known_residual_failure_is_recorded(self):
        """Documented limit: a lexical gate still answers this trap. If this ever starts abstaining, update the README."""
        self.assertFalse(self.fixed.answer("What is the delivery cost for orders under 50 dollars with standard delivery?")["abstained"])


class Grading(unittest.TestCase):
    def test_negated_phrase_is_not_a_match(self):
        self.assertFalse(phrase_ok("You can NOT return items within 30 days.", "30 days"))

    def test_phrase_with_unrelated_earlier_negation_is_a_match(self):
        self.assertTrue(phrase_ok("Perishable goods cannot be returned but can be refunded if they arrive damaged.", "refunded"))

    def test_phrase_that_itself_contains_negation(self):
        self.assertTrue(phrase_ok("We cannot guarantee a nut-free environment.", "cannot guarantee"))

    def test_wilson_interval(self):
        self.assertEqual(wilson(0, 0), (None, None))
        lo, hi = wilson(8, 8)
        self.assertAlmostEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.676, places=2)

    def test_empty_stratum_does_not_divide_by_zero(self):
        only_answerable = [g for g in TUNED if g[1] is not None]
        m = run(RAG(DOCS), only_answerable)
        self.assertEqual(m["unanswerable_abstained"], (0, 0))
        self.assertEqual(fmt(0, 0), "0/0")


class Doctor(unittest.TestCase):
    def make(self, files):
        d = tempfile.mkdtemp(prefix="doctor-")
        for name, text in files.items():
            os.makedirs(os.path.dirname(os.path.join(d, name)), exist_ok=True)
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(text)
        return d

    GOOD_README = "---\ntitle: x\nsdk: gradio\napp_file: app.py\n---\n# x\n"

    def test_broken_fixture_reports_all_causes(self):
        self.assertEqual(codes(diagnose(os.path.join(ROOT, "fixtures", "colab_style"))), {"D1", "D2", "D3", "D4", "D5", "D6"})

    def test_ready_fixture_is_clean(self):
        self.assertEqual(diagnose(os.path.join(ROOT, "fixtures", "spaces_ready")), [])

    def test_missing_app_is_reported(self):
        self.assertEqual(diagnose(os.path.join(ROOT, "data"))[0][0], "D0")

    def test_bare_launch_is_clean(self):
        d = self.make({"app.py": "import gradio as gr\ndemo = gr.Interface(None)\ndemo.launch()\n",
                       "requirements.txt": "gradio==4.44.1\n", "README.md": self.GOOD_README})
        self.assertEqual(diagnose(d), [])

    def test_wrong_host_and_port_hidden_in_comment_still_flagged(self):
        d = self.make({"app.py": 'demo.launch(server_name="127.0.0.1", server_port=5000)  # host="0.0.0.0" 7860\n',
                       "requirements.txt": "gradio==4.44.1\n", "README.md": self.GOOD_README})
        self.assertEqual([c for c, _ in diagnose(d)], ["D2", "D2"])

    def test_commented_out_bad_code_is_ignored(self):
        d = self.make({"app.py": '# demo.launch(server_name="127.0.0.1", server_port=5000)\n# OPENAI_KEY = "sk-proj-ABCDEFGHIJKLMNOP"\ndemo.launch()\n',
                       "requirements.txt": "gradio==4.44.1\n", "README.md": self.GOOD_README})
        self.assertEqual(diagnose(d), [])

    def test_secret_in_oddly_named_variable_flagged(self):
        d = self.make({"app.py": 'OPENAI_KEY = "sk-proj-ABCDEFGHIJKLMNOP"\ndemo.launch()\n',
                       "requirements.txt": "gradio==4.44.1\n", "README.md": self.GOOD_README})
        self.assertIn("D3", codes(diagnose(d)))

    def test_requirement_parsing(self):
        self.assertEqual(check_requirements("-r base.txt\n--index-url https://x\ngit+https://x/y.git\n# c\n"), [])
        self.assertEqual(codes(check_requirements("gradio>=4  # == pin later\n")), {"D1"})
        self.assertIn("loosely", check_requirements("gradio>=4\n")[0][1])

    def test_write_mode_open_is_not_a_missing_input(self):
        d = self.make({"app.py": 'open("out.txt", "w").write("x")\ndemo.launch()\n',
                       "requirements.txt": "gradio==4.44.1\n", "README.md": self.GOOD_README})
        self.assertNotIn("D5", codes(diagnose(d)))

    def test_nonsense_front_matter_flagged(self):
        d = self.make({"app.py": "demo.launch()\n", "requirements.txt": "gradio==4.44.1\n",
                       "README.md": "---\nsdk: nonsense\napp_file: absent.py\n"})
        self.assertEqual(codes(diagnose(d)), {"D6"})
        d2 = self.make({"app.py": "demo.launch()\n", "requirements.txt": "gradio==4.44.1\n",
                        "README.md": "---\nsdk: nonsense\napp_file: absent.py\n---\n"})
        self.assertEqual(len([1 for c, _ in diagnose(d2) if c == "D6"]), 2)


if __name__ == "__main__":
    unittest.main()
