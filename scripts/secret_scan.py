"""Local secret and private-data scan for this repository. Standard library only,
nothing leaves this PC, and matched values are never printed (only file, line and kind).

    python scripts/secret_scan.py               # every file git would commit (or the whole tree without git)
    python scripts/secret_scan.py --staged      # only staged files (used by the pre-commit hook)
    python scripts/secret_scan.py PATH ...      # specific files or folders

Exit status: 0 = nothing found, 1 = findings, 2 = usage error.

It complements, not replaces, a dedicated scanner such as gitleaks
(see .gitleaks.toml). Test fixtures use deliberately fake tokens: a token
that contains an obvious sequential run (abcdefgh, 12345678) or a word such
as "test", "fake" or "example" is treated as a fake and ignored. Add
"secret-scan: allow" to a line to silence a reviewed false positive.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", "data"}
BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".ttf", ".otf", ".woff", ".woff2", ".onnx", ".wav"}
FAKE_RUNS = ("abcdefgh", "01234567", "12345678", "xxxxxxxx")
FAKE_WORDS = ("test", "fake", "dummy", "example", "placeholder", "redacted", "your-")

RULES: list[tuple[str, re.Pattern[str]]] = [
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenRouter API key", re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}")),
    ("OpenAI-style API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}")),
    ("Groq API key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}")),
    ("Google OAuth client secret", re.compile(r"\bGOCSPX-[A-Za-z0-9_\-]{10,}")),
    ("Google OAuth token", re.compile(r"\bya29\.[A-Za-z0-9_\-]{20,}")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[abpors]-[A-Za-z0-9\-]{10,}")),
    ("Tailscale key", re.compile(r"\btskey-[A-Za-z0-9\-]{10,}")),
    ("JSON Web Token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("URL with embedded credentials", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s/:@\"'<>]+:[^\s/@\"'<>]{3,}@[^\s/\"'<>]+")),
    ("hard-coded secret assignment", re.compile(
        r"(?i)\b(?:api[_-]?key|secret|password|passwd|access[_-]?token|auth[_-]?token)\b[\"']?\s*[:=]\s*[\"']"
        r"(?=[A-Za-z0-9_\-+/=.]*\d)(?=[A-Za-z0-9_\-+/=.]*[A-Za-z])([A-Za-z0-9_\-+/=.]{24,})[\"']")),
    ("Windows user-profile path", re.compile(
        r"(?i)\b[A-Z]:[\\/]{1,2}Users[\\/]{1,2}(?!(?:me|you|user|username|public|default|example|name)\b)"
        r"[A-Za-z0-9._\-]+")),
]
# A real PEM block: the header followed by a base64 body on the next lines
# (code that merely names the header, like a format check, doesn't match).
PEM_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----\r?\n[A-Za-z0-9+/=\r\n]{64,}")
# Files whose contents must never be committed, whatever they contain.
FORBIDDEN_NAMES = re.compile(
    r"(?i)(^|/)(\.env(\..+)?|.*\.(pem|key|p12|pfx|dpapi|keystore|jks|db|sqlite3?|log)|id_(rsa|ed25519)[^/]*"
    r"|gmail_(token|client_secret)[^/]*\.json)$")
FORBIDDEN_OK = {".env.example"}


def is_fake(token: str) -> bool:
    low = token.lower()
    return any(run in low for run in FAKE_RUNS + FAKE_WORDS)


def scan_text(rel: str, text: str) -> list[str]:
    findings = [f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: private key" for m in PEM_KEY.finditer(text)]
    for lineno, line in enumerate(text.splitlines(), 1):
        if "secret-scan: allow" in line:
            continue
        for kind, rx in RULES:
            for m in rx.finditer(line):
                if is_fake(m.group(m.lastindex or 0)):
                    continue
                findings.append(f"{rel}:{lineno}: {kind}")
    return findings


def git_files(staged: bool) -> list[str] | None:
    cmd = (["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"] if staged
           else ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"])
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def _display(p: Path) -> str:
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def walk(base: Path) -> list[str]:
    files = []
    for p in base.rglob("*") if base.is_dir() else [base]:
        if p.is_file() and not (set(p.relative_to(base).parts[:-1] if base.is_dir() else ()) & SKIP_DIRS):
            files.append(_display(p))
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--staged", action="store_true")
    args = ap.parse_args(argv)
    if args.paths:
        files = [f for p in args.paths for f in walk(p.resolve())]
    else:
        files = git_files(args.staged)
        if files is None:
            if args.staged:
                print("secret_scan: --staged needs git", file=sys.stderr)
                return 2
            files = walk(ROOT)
    findings: list[str] = []
    for rel in files:
        path = ROOT / rel
        if not path.is_file():
            continue
        if FORBIDDEN_NAMES.search(rel) and Path(rel).name not in FORBIDDEN_OK:
            findings.append(f"{rel}: file type that must not be committed (local secret, key, database or log)")
            continue
        if path.suffix.lower() in BINARY_EXT:
            continue
        data = path.read_bytes()
        if b"\x00" in data[:4096]:
            continue
        findings.extend(scan_text(rel, data.decode("utf-8", "replace")))
    for f in findings:
        print(f)
    print(f"secret_scan: {len(files)} file(s) checked, {len(findings)} finding(s)", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
