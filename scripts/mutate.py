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
