"""Environment protocol — implement this to connect any sandbox to Foundry."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from evoforge.core.types import ToolCall, ToolResult


class EnvironmentState(dict):  # type: ignore[type-arg]
    """Snapshot of the current world state. Serializable dict subclass."""


class EnvironmentSnapshot(dict):  # type: ignore[type-arg]
    """Full serialized environment state for replay / reproducibility."""


class GoalCheckResult:
    def __init__(self, score: float, sub_goals_met: dict[str, bool]) -> None:
        self.score = score                    # 0.0 - 1.0
        self.sub_goals_met = sub_goals_met    # per-sub-goal pass/fail


class FailureConfig:
    def __init__(
        self,
        failure_type: str,          # rate_limit | invalid_param | timeout | partial
        target_tool: str | None = None,
        trigger_after_n_calls: int = 1,
        probability: float = 1.0,
    ) -> None:
        self.failure_type = failure_type
        self.target_tool = target_tool
        self.trigger_after_n_calls = trigger_after_n_calls
        self.probability = probability


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """
    Protocol that any sandbox environment must implement to integrate with Foundry.

    Built-in implementation: SyntheticEnv (auto-generated from tool manifest).
    External implementations: DockerSandboxConnector, HTTPSandboxConnector,
    MCPConnector, or any custom class implementing this interface.

    Example::

        class MyEnv(EnvironmentProtocol):
            def reset(self, seed):     ...
            def step(self, action):    ...
            def get_state(self):       ...
            def check_goal(self, gold): ...
            def check_milestone(self, milestone): ...
            def inject_failure(self, config): ...
            def snapshot(self):        ...
            def restore(self, snap):   ...
            def close(self):           ...
    """

    def reset(self, seed: Any) -> EnvironmentState:
        """Initialize environment for a specific eval case. Must be seedable."""
        ...

    def step(self, action: ToolCall) -> ToolResult:
        """Execute one tool call, update world state, return result."""
        ...

    def get_state(self) -> EnvironmentState:
        """Snapshot current world state for milestone + goal checking."""
        ...

    def check_goal(self, gold: Any) -> GoalCheckResult:
        """Check whether current state satisfies the gold final state."""
        ...

    def check_milestone(self, milestone: Any) -> bool:
        """Check if a specific trajectory milestone has been reached."""
        ...

    def inject_failure(self, config: FailureConfig) -> None:
        """Inject controlled failure for robustness testing."""
        ...

    def snapshot(self) -> EnvironmentSnapshot:
        """Serialize full state for replay and debugging."""
        ...

    def restore(self, snapshot: EnvironmentSnapshot) -> None:
        """Restore from a snapshot for reproducible re-runs."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...
