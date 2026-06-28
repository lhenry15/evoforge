"""HTTP sandbox connector stub (V0.2)."""

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


class HTTPSandboxConnector:
    """
    Connect any REST API sandbox to Foundry via the EnvironmentProtocol.

    V0.2 feature — stub only in V0.1.

    Example::

        sdk.env.connect(
            HTTPSandboxConnector(base_url="http://my-sandbox:8080")
        )
    """

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        self.base_url = base_url

    def reset(self, seed: Any) -> EnvironmentState:
        raise NotImplementedError("HTTPSandboxConnector available in V0.2")

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
