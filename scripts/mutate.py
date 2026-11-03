"""Mutation check. Plants one small defect at a time in src/ (on a temporary copy), runs the test suite and
requires it to FAIL. Exit code 1 if any mutation survives or its pattern no longer matches exactly once.
Run from the lab root: python3 scripts/mutate.py. The table below is the exact record of what is planted."""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MUTATIONS = [
    ("rag.py", "answerability gate removed", "if score < self.threshold or coverage < self.min_coverage:", "if score < self.threshold:"),
    ("rag.py", "similarity threshold removed", "if score < self.threshold or coverage < self.min_coverage:", "if coverage < self.min_coverage:"),
    ("rag.py", "title boost removed", "vec[tok] += 2", "vec[tok] += 0"),
    ("rag.py", "overlapping sliding windows", "range(0, len(sents), window)", "range(0, len(sents))"),
    ("rag.py", "trailing dots no longer stripped", 'w = w.strip(".")', "pass"),
    ("rag.py", "empty-corpus guard removed", "if not hits:  # empty corpus", "if False:  # empty corpus"),
    ("rag.py", "window validation removed", "if not isinstance(window, int) or window < 1:", "if False:"),
    ("rag.py", "generic question words kept", "if t not in GENERIC}", "if True}"),
    ("evaluate.py", "negation guard removed", 'return False  # "not <expected phrase>" is a contradiction, not a match', "pass"),
    ("doctor.py", "comments no longer stripped", "app = strip_comments(raw_app)", "app = raw_app"),
    ("doctor.py", "write-mode opens treated as inputs", 'or set(mode) & set("wax+")', ""),
    ("doctor.py", "loose pins not reported", 'else unpinned).append(line)', 'else unpinned).append(line)\n        loose.clear()'),
    ("doctor.py", "front matter sdk check removed", 'if fm.get("sdk") not in SDKS:', "if False:"),
    ("doctor.py", "app_file existence check removed", ' or not os.path.exists(os.path.join(root, fm["app_file"]))', ""),
    ("doctor.py", "host check removed", 'if host and host.group(1).strip("\\"\'") != "0.0.0.0":', "if False:"),
    ("doctor.py", "port check removed", 'if port and not re.search(r"\\b7860\\b|environ", port.group(1)):', "if False:"),
    ("doctor.py", "secret check removed", 'issues.append(("D3", "secret hard-coded in source"))', "pass"),
    ("doctor.py", "Colab check removed", 'issues.append(("D4", "Colab-only import or path"))', "pass"),
    ("doctor.py", "missing-file check removed", 'issues.append(("D5", f"file opened for reading but not in repo: {path}"))', "pass"),
    # RAGv2 (real-data fix), added 2026-11-03
    ("rag2.py", "v2: plural y-rule removed", 'return w[:-3] + "y"', "return w"),
    ("rag2.py", "v2: -ing rule removed", "return w[:-3]\n    if n > 4 and w.endswith(\"ed\")", "return w\n    if n > 4 and w.endswith(\"ed\")"),
    ("rag2.py", "v2: stop words not removed", "if w not in STOP2]\n\n\ndef question_type", "if True]\n\n\ndef question_type"),
    ("rag2.py", "v2: type bonus removed", "if qtype and type_evidence(qtype, text, question):", "if False:"),
    ("rag2.py", "v2: paragraph context removed", 'sc = sv / s_max + self.r["w_par"] * ps.get(di, 0.0) / p_max', "sc = sv / s_max"),
    ("rag2.py", "v2: title boost removed", "self.par_tokens.append(ptoks + ttoks + ttoks)", "self.par_tokens.append(ptoks)"),
    ("rag2.py", "v2: unseen words count as zero idf", "self.idf_unseen = math.log(1 + (self.n + 0.5) / 0.5)", "self.idf_unseen = 0.0"),
    ("rag2.py", "v2: margin always zero", "margin = (sc1 - sc2) / sc1 if sc1 > 0 else 0.0", "margin = 0.0"),
    ("rag2.py", "v2: missing-name feature removed", "ent_miss = (sum(e not in ptoks for e in ents) / len(ents)) if ents else 0.0", "ent_miss = 0.0"),
    ("rag2.py", "v2: missing-number feature removed", "num_miss = (sum(t not in ptoks for t in nums) / len(nums)) if nums else 0.0", "num_miss = 0.0"),
    ("rag2.py", "v2: missing-type feature removed", "type_miss = 0.0 if type_evidence(qtype, text, question) else 1.0", "type_miss = 0.0"),
    ("rag2.py", "v2: abstain rule removed", 'if p < self.gate["threshold"]:', "if False:"),
    ("rag2.py", "v2: sentence word-pair feature removed", "bigram_sent = (sum(p in sp for p in pairs) / len(pairs)) if pairs else cov_sent", "bigram_sent = 0.0"),
    ("rag2.py", "v2: compactness feature removed", "compact = matched_n / best_span", "compact = 1.0"),
    ("rag2.py", "v2: rare-miss feature removed", "rare_miss = (max(miss_idf) / top) if miss_idf else 0.0", "rare_miss = 0.0"),
    ("rag2.py", "v2: number evidence always true", "return any(w.isdigit() or w in NUMWORDS for w in low)", "return True"),
    ("rag2.py", "v2: month names not evidence of a date", 'or w in MONTHS or w in ("century", "decade")', 'or w in ("century", "decade")'),
    ("rag2.py", "v2: question words allowed as names", "and w.lower() not in qwords and w.lower() not in STOP2:", "and w.lower() not in STOP2:"),
    ("rag2.py", "v2: first word of question counted as a name", "if i > 0 and w[0].isupper() and w.lower() not in STOP2]", "if w[0].isupper() and w.lower() not in STOP2]"),
    ("rag2.py", "v2: sentence coverage reads the paragraph", "cov_sent = sum(idf(t) for t in qterms if t in stoks) / total", "cov_sent = sum(idf(t) for t in qterms if t in ptoks) / total"),
    ("rag2.py", "v2: standardisation ignores the mean", 'z += g["w"][j] * (x[j] - g["mean"][j]) / g["std"][j]', 'z += g["w"][j] * x[j] / g["std"][j]'),
    ("rag2.py", "v2: sentence ties no longer broken by order", "scored.sort(key=lambda x: (-x[0], x[1]))", "scored.sort(key=lambda x: -x[0])"),
]


def suite_passes(d):
    r = subprocess.run([sys.executable, "-B", "-I", "-m", "unittest", "discover", "-s", "tests"], cwd=d,
                       capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return r.returncode == 0


def main():
    tmp = tempfile.mkdtemp(prefix="mutate-rag-")
    work = os.path.join(tmp, "lab")
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(".git", "__pycache__", "evidence"))
    if not suite_passes(work):
        print("baseline suite FAILS; refusing to run mutations")
        sys.exit(2)
    bad = 0
    for i, (fname, desc, old, new) in enumerate(MUTATIONS, 1):
        path = os.path.join(work, "src", fname)
        with open(path, encoding="utf-8") as f:
            original = f.read()
        if original.count(old) != 1:
            print(f"M{i:02d} SKIPPED (pattern matches {original.count(old)} times): {desc}")
            bad += 1
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(original.replace(old, new))
        killed = not suite_passes(work)
        with open(path, "w", encoding="utf-8") as f:
            f.write(original)
        print(f"M{i:02d} {'KILLED  ' if killed else 'SURVIVED'} {fname}: {desc}")
        bad += 0 if killed else 1
    print(f"restored baseline passes: {suite_passes(work)}")
    print(f"{len(MUTATIONS) - bad} of {len(MUTATIONS)} mutations killed")
    shutil.rmtree(tmp)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
