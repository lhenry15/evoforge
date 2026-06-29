"""
EvoForge Example: Code Review Agent

An agent that reviews Python code and outputs structured JSON feedback.
Run the full evolution loop with:

    evoforge run examples/code_agent.py --cycles 4

Prerequisites:
    ollama pull qwen2.5:3b
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openai import OpenAI

import evoforge as foundry
from evoforge import AgentConfig, ModelConfig, ModelHost, Message

# ── SDK setup ─────────────────────────────────────────────────────────────────

sdk = foundry.init(
    task_spec=(
        "A Python code review assistant. Analyzes code for bugs, security issues, "
        "and style problems. Provides suggestions for improvement."
    ),
)

# ── Agent ─────────────────────────────────────────────────────────────────────

@sdk.agent(
    tools=[
        "review_code(code) -> review with issues and suggestions",
        "explain_code(code) -> plain English explanation",
    ],
    config=AgentConfig(
        system_prompt="You are a Python code reviewer. Analyze code for correctness, security, and style.",
        model=ModelConfig(id="qwen2.5:3b", host=ModelHost.LOCAL),
    ),
)
def code_agent(messages: list[Message]) -> str:
    """Code review agent powered by local Ollama."""
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    sys_prompt = code_agent._foundry_agent_config.system_prompt
    skills = code_agent._foundry_agent_config.skill_prompts
    if skills:
        sys_prompt += "\n\nRULES:\n" + "\n".join(f"- {v}" for v in skills.values())

    oai = [{"role": "system", "content": sys_prompt}]
    for m in messages:
        oai.append({"role": m.role, "content": m.content})

    resp = client.chat.completions.create(
        model="qwen2.5:3b", messages=oai, max_tokens=500,
    )
    return resp.choices[0].message.content or ""
