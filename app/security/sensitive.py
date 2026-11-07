"""Deterministic checks shared by the computer-use tools, independent of
the LLM:

- `consequential_reason(label)`: does a UI label (a button name, a CSS
  selector, a hotkey) look like an action that commits something -- send,
  submit, delete, pay, install...? Used to decide which clicks/keys need the
  user's confirmation (see the desktop and browser tools).
- `sensitive_path_reason(path)`: is a file a credential store (SSH keys,
  .env files, browser password databases...)? Such files are never read
  silently.
- `redact_secrets(text)`: masks well-known secret formats (API keys, private
  keys, bearer tokens) in tool output before it reaches the LLM, i.e. before
  it can leave this PC.
"""

from __future__ import annotations

import re
from pathlib import PurePath

_CONSEQUENTIAL_WORDS = (
    # English
    "send",
    "send now",
    "submit",
    "delete",
    "delete permanently",
    "remove",
    "erase",
    "wipe",
    "uninstall",
    "install",
    "buy",
    "buy now",
    "purchase",
    "pay",
    "pay now",
    "checkout",
    "check out",
    "place order",
    "place your order",
    "order now",
    "transfer",
    "publish",
    "post",
    "tweet",
    "upload",
    "confirm",
    "discard",
    "empty recycle bin",
    "reset",
    "factory reset",
    "overwrite",
    "unsubscribe",
    "deactivate",
    "close account",
    "change password",
    "sign out everywhere",
    # Hebrew
    "שלח",
    "שליחה",
    "לשלוח",
    "לשליחה",
    "מחק",
    "מחיקה",
    "למחוק",
    "הסר",
    "הסרה",
    "קנה",
    "קנייה",
    "רכישה",
    "לרכישה",
    "תשלום",
    "לתשלום",
    "שלם",
    "העבר",
    "העברה",
    "פרסם",
    "פרסום",
    "אשר",
    "אישור",
    "הזמן",
    "הזמנה",
    "התקן",
    "התקנה",
)
_CONSEQUENTIAL_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(w) for w in sorted(_CONSEQUENTIAL_WORDS, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def consequential_reason(*labels: str | None) -> str | None:
    """A short reason if any label names a committing action, else None.
    Whole words only, so "Send" / "btn-send" / "sendButton" match but
    "Sender", "Deleted Items" or "Posts" (navigation) don't."""
    for label in labels:
        if not label:
            continue
        normalized = re.sub(r"[_\-#.\[\]=\"':]+", " ", _CAMEL_RE.sub(" ", label))
        match = _CONSEQUENTIAL_RE.search(normalized)
        if match:
            return f"'{label.strip()[:80]}' looks like a committing action ({match.group(1).lower()})."
    return None


# ------------------------------------------------------------- credentials
_SENSITIVE_NAMES = {
    ".env",
    ".netrc",
    "_netrc",
    ".git-credentials",
    ".pgpass",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "known_hosts",
    "login data",
    "login data for account",
    "cookies",
    "web data",
    "local state",
    "key3.db",
    "key4.db",
    "logins.json",
    "cert9.db",
    "gmail_token.json",
    "gmail_client_secret.json",
    "devices.json",
    "token.json",
    "ca.key",
    "server.key",
    "keys.dpapi",
    # package-registry / CLI tokens and other plaintext secret files
    ".npmrc",
    ".pypirc",
    ".yarnrc.yml",
    ".htpasswd",
    ".vault-token",
    ".credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "wallet.dat",
    "sitemanager.xml",
    "recentservers.xml",
}
_SENSITIVE_SUFFIXES = {".pem", ".key", ".pfx", ".p12", ".kdbx", ".keystore", ".jks", ".ppk", ".asc", ".gpg"}
_SENSITIVE_DIRS = {
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".kube",
    ".docker",
    "microsoft\\credentials",
    "protect",
    ".config\\gh",
    ".claude",
    ".omniroute",
}


def sensitive_path_reason(path: str | PurePath) -> str | None:
    """A reason if `path` looks like a credential/secret store, else None."""
    pure = PurePath(str(path))
    name = pure.name.lower()
    if name in _SENSITIVE_NAMES or name.startswith(".env."):
        return f"'{pure.name}' typically holds credentials or tokens."
    if pure.suffix.lower() in _SENSITIVE_SUFFIXES:
        return f"'{pure.name}' looks like a key or certificate store."
    lowered = str(pure).lower().replace("/", "\\")
    for directory in _SENSITIVE_DIRS:
        if f"\\{directory}\\" in lowered + "\\" or lowered.endswith(f"\\{directory}"):
            return f"'{pure}' is inside a credential directory."
    return None


_SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(-----END [A-Z ]*PRIVATE KEY-----|$)", re.DOTALL),
    re.compile(r"\b(sk|rk|pk)-[A-Za-z0-9_\-]{20,}"),  # OpenAI/Anthropic/Stripe-style keys
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),  # Google API keys
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),  # GitHub tokens
    re.compile(r"\bgsk_[A-Za-z0-9]{30,}"),  # Groq keys
    re.compile(r"\bnpm_[A-Za-z0-9]{30,}"),  # npm tokens
    re.compile(r"\bpypi-[A-Za-z0-9_\-]{40,}"),  # PyPI tokens
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),  # Slack tokens
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key ids
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWTs
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{20,}"),
    re.compile(
        # NAME=value / name: value anywhere (env files, logs, URL queries); the
        # value runs to the next whitespace, quote, '&' or ';' and must look
        # like a secret (6+ chars, not just a number -- "max_tokens: 1200").
        r"(?i)((?<![\w])[\w.\-]*(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?key|token|private[_-]?key)"
        r"[\w.\-]*[ \t]*[=:][ \t]*[\"']?)(?![\d.,]+(?:[\s\"'&;]|$))([^\s\"'&;]{6,})"
    ),
]


def redact_secrets(text: str) -> str:
    """Mask secret-looking substrings; everything else is unchanged."""
    if not text:
        return text
    for pattern in _SECRET_PATTERNS[:-1]:
        text = pattern.sub("[REDACTED]", text)
    return _SECRET_PATTERNS[-1].sub(lambda m: m.group(1) + "[REDACTED]", text)
