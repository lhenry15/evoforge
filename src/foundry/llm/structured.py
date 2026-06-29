"""Structured generation helpers — robust JSON from LLMs, even small ones.

Two layers of defense:
  1. Schema-constrained decoding when the pool supports it (Ollama ``format=schema``).
  2. A robust fallback parser that repairs the common small-model failure modes
     (markdown fences, trailing commas, and — critically — double-wrapped arrays
     like ``[[{...}]]`` that 3B models frequently emit).
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Optional


def _strip_fences(raw: str) -> str:
    cleaned = raw.strip()
    if "```" in cleaned:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if m:
            return m.group(1).strip()
    return cleaned


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[\]}])", r"\1", text)


def extract_json(raw: str) -> Optional[Any]:
    """Best-effort parse of a JSON value from noisy model output."""
    if not raw:
        return None
    cleaned = _strip_fences(raw)

    # Direct parse first.
    for candidate in (cleaned, _remove_trailing_commas(cleaned)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass

    # Fall back to the largest object/array substring.
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, cleaned, re.DOTALL)
        if m:
            snippet = _remove_trailing_commas(m.group())
            try:
                return json.loads(snippet)
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def flatten_dicts(data: Any) -> list[dict[str, Any]]:
    """Recursively flatten nested lists into a flat list of dicts.

    Handles the frequent 3B failure where the model double-wraps:
    ``[[{...}], [{...}]]`` or ``[[{...}, {...}]]`` -> ``[{...}, {...}]``.
    """
    out: list[dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            out.append(node)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(data)
    return out


def coerce_records(parsed: Any, key: Optional[str] = None) -> list[dict[str, Any]]:
    """Coerce a parsed structure into a flat list of record dicts.

    Accepts ``{key: [...]}``, ``[...]``, ``[[...]]``, or a single ``{...}``.
    """
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        if key and isinstance(parsed.get(key), list):
            return flatten_dicts(parsed[key])
        # A wrapper object with a single list value (whatever the key).
        list_values = [v for v in parsed.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return flatten_dicts(list_values[0])
        return [parsed]
    return flatten_dicts(parsed)


def coerce_strings(parsed: Any, key: Optional[str] = None) -> list[str]:
    """Coerce a parsed structure into a flat list of strings."""
    if parsed is None:
        return []
    if isinstance(parsed, dict) and key and isinstance(parsed.get(key), list):
        parsed = parsed[key]
    out: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            # Pull the most string-like value (covers {"message": "..."}).
            for v in node.values():
                if isinstance(v, str):
                    out.append(v)
                    break

    _walk(parsed)
    return [s for s in out if s.strip()]


def generate_structured(
    pool: Any,
    prompt: str,
    schema: dict[str, Any],
    system: str = "You are a helpful AI assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Optional[Any]:
    """Generate and parse JSON from a pool, preferring schema-constrained decoding.

    Returns the parsed object (dict/list) or ``None`` if nothing parseable came back.
    """
    # 1. Native structured output (e.g. Ollama format=schema).
    gen_json = getattr(pool, "generate_json", None)
    if callable(gen_json):
        try:
            result = gen_json(
                prompt, schema=schema, system=system,
                temperature=temperature, max_tokens=max_tokens,
            )
            if inspect.isawaitable(result):
                raise RuntimeError(
                    "generate_structured received an async pool in sync mode."
                )
            if result is not None:
                return result
        except RuntimeError:
            raise
        except Exception:
            pass  # fall through to text + robust parse

    # 2. Plain generation + robust parse.
    raw = pool.generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
    if inspect.isawaitable(raw):
        raise RuntimeError("generate_structured received an async pool in sync mode.")
    return extract_json(str(raw))
