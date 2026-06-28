"""MCP (Model Context Protocol) connector stub (V0.2)."""

from __future__ import annotations

from typing import Any
from foundry.environment.protocol import (
    EnvironmentProtocol,
    EnvironmentSnapshot,
    EnvironmentState,
    FailureConfig,
    GoalCheckResult,
)
from foundry.core.types import ToolCall, ToolResult


class MCPConnector:
    """
    Connect any MCP server as an environment for Foundry eval.

    V0.2 feature — stub only in V0.1.

    Example::

        sdk.env.connect(MCPConnector(server_url="http://localhost:8080"))
    """

    def __init__(self, server_url: str, **kwargs: Any) -> None:
        self.server_url = server_url

    def reset(self, seed: Any) -> EnvironmentState:
        raise NotImplementedError("MCPConnector available in V0.2")

    def step(self, action: ToolCall) -> ToolResult:
        raise NotImplementedError

    def get_state(self) -> EnvironmentState:
        raise NotImplementedError

    def check_goal(self, gold: Any) -> GoalCheckResult:
        raise NotImplementedError

    def check_milestone(self, milestone: Any) -> bool:
        raise NotImplementedError

    def inject_failure(self, config: FailureConfig) -> None:
        raise NotImplementedError

    def snapshot(self) -> EnvironmentSnapshot:
        raise NotImplementedError

    def restore(self, snapshot: EnvironmentSnapshot) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass
