"""
EvoForge Example: Flight Booking Agent

A simple agent that searches for flights and books them.
Run the full evolution loop with:

    evoforge run examples/flight_agent.py --cycles 4

Prerequisites:
    ollama pull qwen2.5:3b
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openai import OpenAI

import foundry
from foundry.core.agent_config import AgentConfig, ModelConfig, ModelHost
from foundry.core.types import Message

# ── SDK setup ─────────────────────────────────────────────────────────────────

sdk = foundry.init(
    task_spec=(
        "A flight booking assistant. Searches for available flights "
        "and books them for passengers. Always confirms price before booking."
    ),
)

# ── Agent ─────────────────────────────────────────────────────────────────────

@sdk.agent(
    tools=[
        "search_flights(origin, destination, date) -> list of flights with prices",
        "book_flight(flight_id, passenger_name) -> booking confirmation with reference",
    ],
    config=AgentConfig(
        system_prompt="You are a helpful flight booking assistant. Always confirm price before booking.",
        model=ModelConfig(id="qwen2.5:3b", host=ModelHost.LOCAL),
    ),
)
def flight_agent(messages: list[Message]) -> str:
    """Flight booking agent powered by local Ollama."""
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    sys_prompt = flight_agent._foundry_agent_config.system_prompt
    skills = flight_agent._foundry_agent_config.skill_prompts
    if skills:
        sys_prompt += "\n\n" + "\n".join(f"[{k}] {v}" for k, v in skills.items())

    oai = [{"role": "system", "content": sys_prompt}]
    for m in messages:
        oai.append({"role": m.role, "content": m.content})

    tools = [
        {"type": "function", "function": {
            "name": "search_flights", "description": "Search available flights",
            "parameters": {"type": "object", "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string"},
            }, "required": ["origin", "destination", "date"]},
        }},
        {"type": "function", "function": {
            "name": "book_flight", "description": "Book a flight for a passenger",
            "parameters": {"type": "object", "properties": {
                "flight_id": {"type": "string"},
                "passenger_name": {"type": "string"},
            }, "required": ["flight_id", "passenger_name"]},
        }},
    ]

    for _ in range(3):
        resp = client.chat.completions.create(
            model="qwen2.5:3b", messages=oai,
            tools=tools, tool_choice="auto", max_tokens=300,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""

        oai.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                           "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                          for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            if tc.function.name == "search_flights":
                result = f"Available flights: UA123 $320, AA456 $289, DL789 $355"
            elif tc.function.name == "book_flight":
                fid = args.get("flight_id", "?")
                pax = args.get("passenger_name", "?")
                result = f"Confirmed: {fid} for {pax}. Ref: BK{abs(hash(fid)) % 10000:04d}. Price: $289."
            else:
                result = "Unknown tool"
            oai.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Could not complete the request."
