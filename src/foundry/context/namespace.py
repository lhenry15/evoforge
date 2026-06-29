"""ContextNamespace — sdk.context interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foundry.core.sdk import FoundrySDK


class ContextNamespace:
    """sdk.context — GroupContext and PerUserContext management (V1.0)."""

    def __init__(self, sdk: "FoundrySDK") -> None:
        self._sdk = sdk

    def load_group_context(self, group_id: str) -> dict[str, Any]:
        raise NotImplementedError("GroupContext management is a V1.0 feature.")

    def bootstrap_from_logs(self, log_source: Any) -> None:
        raise NotImplementedError("Chat log bootstrap is a V1.0 feature.")

