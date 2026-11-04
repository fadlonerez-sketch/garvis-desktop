"""Structured logging setup for Garvis.

Logs are emitted as single-line JSON so they can be greppable/parseable.
Callers must never pass secrets (tokens, passwords, OAuth credentials) into
log fields -- see `redact()` for the small set of keys we defensively mask
if they slip through.
"""
from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

_EXCERPT_MAX_CHARS = 120

_SENSITIVE_KEYS = {
    "password",
    "token",
    "access_token",
    "refresh_token",
    "client_secret",
    "authorization",
    "api_key",
    "api-key",
    "apikey",
    "secret",
    "cookie",
    "credential",
    "private_key",
}


def redact(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` with sensitive-looking keys masked."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if any(marker in key.lower() for marker in _SENSITIVE_KEYS):
            out[key] = "***REDACTED***"
        elif isinstance(value, dict):
            out[key] = redact(value)
        else:
            out[key] = value
    return out


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(redact(extra))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(log_level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure the root Garvis logger. Safe to call multiple times."""
    root = logging.getLogger("garvis")
    root.setLevel(log_level.upper())
    root.handlers.clear()

    formatter = JsonFormatter()

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "garvis.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"garvis.{name}")


def log_extra(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Log ``message`` with structured ``fields`` attached (auto-redacted)."""
    logger.log(level, message, extra={"extra_fields": fields})


def content_fingerprint(text: str) -> dict[str, Any]:
    """Diagnostic-only summary of ``text``, safe to log: never the full
    text itself, just enough to tell whether two log lines carried the
    exact same content (``content_hash``) without writing that content out
    in full. Used at the LLM-provider and assistant-message-dispatch log
    points so duplicate/repeated output can be diagnosed from logs alone --
    see app/llm/groq.py, app/ui/agent_bridge.py.

    The hash is a deliberately short, non-cryptographic-strength prefix
    (collisions are an acceptable, cheap tradeoff for a debugging aid, not
    a security control) -- identical text always yields the identical hash
    so entries can be correlated across log lines/files.
    """
    excerpt = " ".join(text.split())[:_EXCERPT_MAX_CHARS]
    return {
        "content_length": len(text),
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
        "content_excerpt": excerpt,
    }
