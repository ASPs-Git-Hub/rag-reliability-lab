"""Deployment doctor: static checks for the usual 'worked in Colab, fails on Hugging Face Spaces' causes.

Static and regex-based: it reads files and never runs the project, so it cannot see runtime errors
and a clean report does not prove the Space will start. Comment stripping is line-based and crude.
Usage: python src/doctor.py <project_dir>
"""
import os
import re
import sys

SDKS = {"gradio", "streamlit", "docker", "static"}


def read(p):
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def strip_comments(code):
    return "\n".join(re.sub(r"(^|\s)#.*$", "", line) for line in code.splitlines())


def check_requirements(reqs):
    issues = []
    unpinned, loose = [], []
    for raw in reqs.splitlines():
        line = raw.split(" #")[0].split("\t#")[0].strip()
        if not line or line.startswith("#") or line.startswith("-") or "git+" in line or "://" in line:
            continue
        if "==" in line:
            continue
        (loose if re.search(r"(>=|~=|<=|>|<)", line) else unpinned).append(line)
    if unpinned:
        issues.append(("D1", "unpinned requirements: " + ", ".join(unpinned)))
    if loose:
        issues.append(("D1", "loosely pinned (not ==): " + ", ".join(loose)))
    return issues


def parse_front_matter(readme):
    m = re.match(r"^---\s*\n(.*?)\n---\s*(\n|$)", readme, re.S)
    if not m:
        return None
    return dict(re.findall(r"^\s*([A-Za-z_]+)\s*:\s*(.+?)\s*$", m.group(1), re.M))


def launch_args(code):
    """Yield the argument text of every `.launch(...)` call, matching parentheses by counting."""
    for m in re.finditer(r"\.launch\(", code):
        i, depth = m.end(), 1
        while i < len(code) and depth:
            depth += {"(": 1, ")": -1}.get(code[i], 0)
            i += 1
        yield code[m.end():i - 1]


def diagnose(root):
    issues = []
    raw_app = read(os.path.join(root, "app.py"))
    if raw_app is None:
        return [("D0", "app.py not found")]
    app = strip_comments(raw_app)

    reqs = read(os.path.join(root, "requirements.txt"))
    if reqs is None:
        issues.append(("D1", "requirements.txt missing"))
    else:
        issues += check_requirements(reqs)

    # D2: host and port. A bare launch() is fine (Spaces sets GRADIO_SERVER_NAME and port itself).
    for args in launch_args(app):
        host = re.search(r"(?:server_name|host)\s*=\s*([\"'][^\"']*[\"'])", args)
        if host and host.group(1).strip("\"'") != "0.0.0.0":
            issues.append(("D2", f"binds to {host.group(1)} instead of 0.0.0.0"))
        port = re.search(r"(?:server_port|port)\s*=\s*([^,)]+(?:\([^)]*\))?)", args)
        if port and not re.search(r"\b7860\b|environ", port.group(1)):
            issues.append(("D2", f"port is {port.group(1).strip()}, not 7860 or read from the environment"))

    # D3: hard-coded secrets
    if re.search(r"(key|secret|token|password)\w*\s*=\s*[\"'][A-Za-z0-9_\-\.]{12,}[\"']", app, re.I) or \
            re.search(r"[\"'](sk-|hf_|ghp_|gho_|xox[bp]-)[A-Za-z0-9_\-]{8,}[\"']", app):
        issues.append(("D3", "secret hard-coded in source"))

    # D4: Colab-only
    if re.search(r"google\.colab|/content/|/drive/", app):
        issues.append(("D4", "Colab-only import or path"))

    # D5: files opened for reading but absent from the repo
    for m in re.finditer(r"open\(\s*[\"']([^\"']+)[\"']\s*(?:,\s*[\"']([^\"']*)[\"'])?", app):
        path, mode = m.group(1), m.group(2) or "r"
        if path.startswith("/") or set(mode) & set("wax+"):
            continue
        if not os.path.exists(os.path.join(root, path)):
            issues.append(("D5", f"file opened for reading but not in repo: {path}"))

    # D6: README front matter
    fm = parse_front_matter(read(os.path.join(root, "README.md")) or "")
    if fm is None:
        issues.append(("D6", "README.md has no closed front matter block (---)"))
    else:
        if fm.get("sdk") not in SDKS:
            issues.append(("D6", f"README front matter sdk must be one of {sorted(SDKS)}, got {fm.get('sdk')!r}"))
        if not fm.get("app_file") or not os.path.exists(os.path.join(root, fm["app_file"])):
            issues.append(("D6", f"README front matter app_file missing or not in repo: {fm.get('app_file')!r}"))
    return issues


if __name__ == "__main__":
    found = diagnose(sys.argv[1])
    for code, msg in found:
        print(f"{code}  {msg}")
    print(f"{len(found)} issue(s)")
    sys.exit(1 if found else 0)
