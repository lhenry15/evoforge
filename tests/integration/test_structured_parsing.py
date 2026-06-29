"""Unit tests for robust structured-JSON parsing (small-model failure modes)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from foundry.llm.structured import (
    coerce_records,
    coerce_strings,
    extract_json,
    flatten_dicts,
    generate_structured,
)


def test_extract_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('[1, 2, 3]') == [1, 2, 3]


def test_extract_strips_markdown_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('```\n[{"x": 1}]\n```') == [{"x": 1}]


def test_extract_repairs_trailing_commas():
    assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}
    assert extract_json('[1, 2, 3,]') == [1, 2, 3]


def test_extract_finds_embedded_json():
    assert extract_json('Sure! Here you go: {"a": 1} hope that helps') == {"a": 1}


def test_extract_returns_none_on_garbage():
    assert extract_json("not json at all") is None
    assert extract_json("") is None


def test_flatten_double_wrapped_array():
    """The core 3B failure: [[{...}]] and [[{...}],[{...}]] -> flat list of dicts."""
    assert flatten_dicts([[{"a": 1}]]) == [{"a": 1}]
    assert flatten_dicts([[{"a": 1}], [{"b": 2}]]) == [{"a": 1}, {"b": 2}]
    assert flatten_dicts([[{"a": 1}, {"b": 2}]]) == [{"a": 1}, {"b": 2}]
    assert flatten_dicts([{"a": 1}, {"b": 2}]) == [{"a": 1}, {"b": 2}]


def test_coerce_records_handles_all_shapes():
    assert coerce_records({"cases": [{"a": 1}]}, key="cases") == [{"a": 1}]
    assert coerce_records([{"a": 1}]) == [{"a": 1}]
    assert coerce_records([[{"a": 1}]]) == [{"a": 1}]            # double-wrapped
    assert coerce_records({"items": [{"a": 1}]}) == [{"a": 1}]   # single list value
    assert coerce_records({"a": 1}) == [{"a": 1}]                # bare object


def test_coerce_strings_handles_all_shapes():
    assert coerce_strings({"messages": ["a", "b"]}, key="messages") == ["a", "b"]
    assert coerce_strings(["a", "b"]) == ["a", "b"]
    assert coerce_strings([["a"], ["b"]]) == ["a", "b"]               # nested
    assert coerce_strings([{"message": "a"}, {"message": "b"}]) == ["a", "b"]
    assert coerce_strings({"messages": []}, key="messages") == []


class _StructuredPool:
    supports_structured = True

    def generate_json(self, prompt, schema, system="", temperature=0.0, max_tokens=512, **kw):
        return {"messages": ["hello", "world"]}

    def generate(self, *a, **k):
        raise AssertionError("should not fall back when generate_json works")


class _PlainPool:
    """No generate_json — must fall back to text + robust parse (double-wrapped)."""

    def generate(self, prompt, system="", temperature=0.0, max_tokens=512, **kw):
        return '[[{"user_message": "book a flight"}]]'


def test_generate_structured_prefers_native_json():
    pool = _StructuredPool()
    result = generate_structured(pool, "p", {"type": "object"})
    assert result == {"messages": ["hello", "world"]}


def test_generate_structured_falls_back_and_repairs():
    pool = _PlainPool()
    result = generate_structured(pool, "p", {"type": "object"})
    # Robust parse returns the double-wrapped structure; coerce_records flattens it.
    assert coerce_records(result) == [{"user_message": "book a flight"}]
