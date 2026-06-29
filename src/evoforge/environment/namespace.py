"""EnvNamespace — sdk.env interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evoforge.core.sdk import FoundrySDK


class EnvNamespace:
    """
    sdk.env — connector lifecycle management.

    Environment protocol implementations are pluggable. This namespace stores the
    active connector and exposes minimal lifecycle hooks used by higher-level
    orchestration.
    """

    _REQUIRED_METHODS = (
        "reset",
        "step",
        "get_state",
        "check_goal",
        "check_milestone",
        "inject_failure",
        "snapshot",
        "restore",
        "close",
    )

    def __init__(self, sdk: "FoundrySDK") -> None:
        self._sdk = sdk
        self._connector: Any | None = None

    def connect(self, connector: Any) -> Any:
        """Register an environment connector implementing the protocol."""
        missing = [name for name in self._REQUIRED_METHODS if not hasattr(connector, name)]
        if missing:
            raise TypeError(
                "Connector does not implement EnvironmentProtocol. "
                f"Missing: {', '.join(missing)}"
            )

        self._connector = connector
        return connector

    @property
    def connector(self) -> Any | None:
        """Return the currently connected environment connector, if any."""
        return self._connector

    def is_connected(self) -> bool:
        """Whether an environment connector is currently attached."""
        return self._connector is not None

    def close(self) -> None:
        """Close and detach the active environment connector."""
        if self._connector is not None:
            self._connector.close()
        self._connector = None
