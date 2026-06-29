"""Local filesystem storage backend (default)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


class LocalStorageBackend:
    """
    Stores all DataRegistry artifacts on the local filesystem.

    Layout::
        base_path/
          eval/
            v1/ ...
            v2/ ...
          train/
            v1/ ...
          skills/
            v1/ ...
          group_context/
            {group_id}/v1/ ...
          metadata.db      ← SQLite for versioning + lineage
    """

    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def read(self, path: str) -> bytes | None:
        full = self.base_path / path
        if not full.exists():
            return None
        return full.read_bytes()

    def write(self, path: str, data: Any) -> None:
        full = self.base_path / path
        full.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, (dict, list)):
            full.write_text(json.dumps(data, indent=2))
        elif isinstance(data, str):
            full.write_text(data)
        else:
            full.write_bytes(data)

    def exists(self, path: str) -> bool:
        return (self.base_path / path).exists()

    def list(self, prefix: str) -> list[str]:
        root = self.base_path / prefix
        if not root.exists():
            return []
        return [str(p.relative_to(self.base_path)) for p in root.rglob("*") if p.is_file()]

    def delete(self, path: str) -> None:
        full = self.base_path / path
        if full.is_dir():
            shutil.rmtree(full)
        elif full.exists():
            full.unlink()
