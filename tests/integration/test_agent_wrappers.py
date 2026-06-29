"""
Integration tests: verify Foundry decorator works correctly with
smolagents, pydantic-ai, and LangChain agents.

These tests do NOT make real LLM calls — they verify that:
  1. @sdk.agent() correctly attaches metadata to each framework's pattern
  2. AgentConfig fields (system_prompt, model, swap_model) are accessible
  3. Evolution engine helpers (_can_fine_tune, _can_auto_promote) work correctly
  4. The thin adapter function pattern wraps each framework cleanly

Run: pytest tests/integration/test_agent_wrappers.py -v
"""

from __future__ import annotations

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import evoforge
from evoforge.core.agent_config import AgentConfig, ModelConfig, ModelHost
from evoforge.core.types import Message
from evoforge.evolution.engine import (
    _can_fine_tune,
    _can_auto_promote,
    _get_system_prompt,
    _get_model_id,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a helpful flight booking assistant."
MODEL_ID = "gpt-4o"
SWAP_CALLED: list[str] = []   # records swap_model calls


def _make_swap_fn(agent_name: str):
    def swap(new_id: str) -> None:
        SWAP_CALLED.append(f"{agent_name}:{new_id}")
    return swap


def _make_sdk():
    return evoforge.init(task_spec="A flight booking assistant for testing.")


def _make_config(with_swap: bool = True) -> AgentConfig:
    return AgentConfig(
        system_prompt=SYSTEM_PROMPT,
        model=ModelConfig(id=MODEL_ID, host=ModelHost.OPENAI),
        swap_model=_make_swap_fn("test") if with_swap else None,
    )


# ── smolagents wrapper tests ──────────────────────────────────────────────────

class TestSmolagentsWrapper:

    def test_decorator_attaches_metadata(self):
        sdk = _make_sdk()

        @sdk.agent(tools=["search_flights", "book_flight"], config=_make_config())
        def smol_agent(messages): ...

        assert hasattr(smol_agent, "_foundry_agent_config")
        assert hasattr(smol_agent, "_foundry_tools")
        assert hasattr(smol_agent, "_foundry_multi_party")
        assert smol_agent._foundry_multi_party is False

    def test_agent_config_fields_accessible(self):
        sdk = _make_sdk()

        @sdk.agent(tools=["search_flights"], config=_make_config())
        def smol_agent(messages): ...

        cfg = smol_agent._foundry_agent_config
        assert cfg.system_prompt == SYSTEM_PROMPT
        assert cfg.model.id == MODEL_ID
        assert cfg.model.host == ModelHost.OPENAI
        assert cfg.swap_model is not None

    def test_can_fine_tune_with_config(self):
        sdk = _make_sdk()

        @sdk.agent(tools=[], config=_make_config())
        def smol_agent(messages): ...

        can_ft, reason = _can_fine_tune(smol_agent)
        assert can_ft is True
        assert reason == ""

    def test_cannot_fine_tune_without_config(self):
        sdk = _make_sdk()

        @sdk.agent(tools=[])
        def smol_agent_black_box(messages): ...

        can_ft, reason = _can_fine_tune(smol_agent_black_box)
        assert can_ft is False
        assert "AgentConfig" in reason

    def test_can_auto_promote_with_swap_fn(self):
        sdk = _make_sdk()

        @sdk.agent(tools=[], config=_make_config(with_swap=True))
        def smol_agent(messages): ...

        can_promo, _ = _can_auto_promote(smol_agent)
        assert can_promo is True

    def test_cannot_auto_promote_without_swap_fn(self):
        sdk = _make_sdk()

        @sdk.agent(tools=[], config=_make_config(with_swap=False))
        def smol_agent(messages): ...

        can_promo, reason = _can_auto_promote(smol_agent)
        assert can_promo is False
        assert "PromotionEvent" in reason

    def test_swap_model_callable(self):
        sdk = _make_sdk()
        swap_log: list[str] = []

        @sdk.agent(
            tools=[],
            config=AgentConfig(
                system_prompt=SYSTEM_PROMPT,
                model=ModelConfig(id=MODEL_ID, host=ModelHost.OPENAI),
                swap_model=lambda new_id: swap_log.append(new_id),
            ),
        )
        def smol_agent(messages): ...

        smol_agent._foundry_agent_config.swap_model("ft:gpt-4o:abc123")
        assert swap_log == ["ft:gpt-4o:abc123"]

    def test_get_system_prompt_helper(self):
        sdk = _make_sdk()

        @sdk.agent(tools=[], config=_make_config())
        def smol_agent(messages): ...

        assert _get_system_prompt(smol_agent) == SYSTEM_PROMPT

    def test_get_system_prompt_returns_none_without_config(self):
        sdk = _make_sdk()

        @sdk.agent(tools=[])
        def smol_agent_black_box(messages): ...

        assert _get_system_prompt(smol_agent_black_box) is None

    def test_get_model_id_helper(self):
        sdk = _make_sdk()

        @sdk.agent(tools=[], config=_make_config())
        def smol_agent(messages): ...

        assert _get_model_id(smol_agent) == MODEL_ID


# ── pydantic-ai wrapper tests ─────────────────────────────────────────────────

class TestPydanticAIWrapper:

    def test_decorator_attaches_metadata(self):
        sdk = _make_sdk()

        @sdk.agent(tools=["search_flights", "book_flight"], config=_make_config())
        def pai_agent(messages): ...

        assert hasattr(pai_agent, "_foundry_agent_config")
        assert pai_agent._foundry_tools == ["search_flights", "book_flight"]

    def test_tool_names_only_accepted(self):
        """pydantic-ai registers tools on the agent object; Foundry accepts names."""
        sdk = _make_sdk()

        @sdk.agent(tools=["search_flights", "book_flight"], config=_make_config())
        def pai_agent(messages): ...

        assert pai_agent._foundry_tools == ["search_flights", "book_flight"]

    def test_full_config_enables_fine_tune(self):
        sdk = _make_sdk()

        @sdk.agent(tools=[], config=_make_config())
        def pai_agent(messages): ...

        can_ft, _ = _can_fine_tune(pai_agent)
        assert can_ft is True

    def test_swap_model_enables_auto_promotion(self):
        sdk = _make_sdk()

        @sdk.agent(tools=[], config=_make_config(with_swap=True))
        def pai_agent(messages): ...

        can_promo, _ = _can_auto_promote(pai_agent)
        assert can_promo is True


# ── LangChain wrapper tests ───────────────────────────────────────────────────

import pytest

langchain_available = True
try:
    from langchain_core.tools import tool as lc_tool
except ImportError:
    langchain_available = False


@pytest.mark.skipif(not langchain_available, reason="langchain_core not installed")
class TestLangChainWrapper:

    def test_decorator_attaches_metadata(self):
        sdk = _make_sdk()

        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def dummy_tool(x: str) -> str:
            """A dummy tool. Args: x: input string"""
            return x

        @sdk.agent(tools=[dummy_tool], config=_make_config())
        def lc_agent(messages): ...

        assert hasattr(lc_agent, "_foundry_agent_config")
        assert len(lc_agent._foundry_tools) == 1
        assert lc_agent._foundry_tools[0].name == "dummy_tool"

    def test_lc_tool_objects_accepted(self):
        """LangChain tool objects (not just names) should be stored as-is."""
        sdk = _make_sdk()

        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def my_tool(q: str) -> str:
            """Search. Args: q: query"""
            return f"results for {q}"

        @sdk.agent(tools=[my_tool], config=_make_config())
        def lc_agent(messages): ...

        assert hasattr(lc_agent._foundry_tools[0], "name")

    def test_full_config_all_helpers(self):
        sdk = _make_sdk()

        @sdk.agent(tools=[], config=_make_config())
        def lc_agent(messages): ...

        assert _can_fine_tune(lc_agent) == (True, "")
        assert _can_auto_promote(lc_agent)[0] is True
        assert _get_system_prompt(lc_agent) == SYSTEM_PROMPT
        assert _get_model_id(lc_agent) == MODEL_ID


# ── Cross-framework consistency tests ────────────────────────────────────────

class TestCrossFramework:

    def test_black_box_behavior_identical_across_frameworks(self):
        """All three frameworks degrade identically without AgentConfig."""
        sdk = _make_sdk()

        @sdk.agent(tools=[])
        def smol(messages): ...

        @sdk.agent(tools=[])
        def pai(messages): ...

        @sdk.agent(tools=[])
        def lc(messages): ...

        for agent_fn in [smol, pai, lc]:
            assert _can_fine_tune(agent_fn)[0] is False
            assert _can_auto_promote(agent_fn)[0] is False
            assert _get_system_prompt(agent_fn) is None
            assert _get_model_id(agent_fn) is None

    def test_group_agent_decorator_sets_multi_party_flag(self):
        sdk = _make_sdk()

        @sdk.group_agent(tools=[], config=_make_config())
        def group_agent(messages, user_id, user_ctx, group_ctx): ...

        assert group_agent._foundry_multi_party is True

    def test_single_agent_not_multi_party(self):
        sdk = _make_sdk()

        @sdk.agent(tools=[])
        def single_agent(messages): ...

        assert single_agent._foundry_multi_party is False

    def test_message_type_works(self):
        """Foundry Message type can represent all roles used by all frameworks."""
        msgs = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Find me a flight to NYC."),
            Message(role="assistant", content="I'll search for that."),
            Message(role="tool", content='{"flights": []}', tool_call_id="call_123"),
        ]
        assert len(msgs) == 4
        assert msgs[1].role == "user"
        assert msgs[3].tool_call_id == "call_123"
