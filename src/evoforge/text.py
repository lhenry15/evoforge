"""Shared text helpers: tokenization, similarity, and tool formatting.

Consolidates logic that was previously copy-pasted across mining, coverage,
synthesis, and factory modules.
"""

from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall((text or "").lower())


def token_set(text: str) -> set[str]:
    """Unique lowercase alphanumeric tokens."""
    return set(tokenize(text))


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two token sets (empty-vs-empty == 1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def jaccard_text(a: str, b: str) -> float:
    """Jaccard similarity between two raw strings."""
    return jaccard(token_set(a), token_set(b))


def format_tools(tools: list[Any] | None) -> str:
    """Render a heterogeneous tool list (strings or objects) for prompts."""
    if not tools:
        return "None"
    out = []
    for t in tools:
        if isinstance(t, str):
            out.append(f"- {t}")
        elif hasattr(t, "name") and hasattr(t, "description"):
            out.append(f"- {t.name}: {t.description}")
        else:
            out.append(f"- {getattr(t, 'name', t)}")
    return "\n".join(out)
