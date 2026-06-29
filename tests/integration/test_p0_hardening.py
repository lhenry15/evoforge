"""Integration tests for P0 hardening changes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from click.testing import CliRunner
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import evoforge
import foundry
from foundry.cli import cli


def test_evoforge_import_alias_exposes_public_api():
    assert callable(evoforge.init)
    assert evoforge.AgentConfig is not None
    assert evoforge.ModelConfig is not None
    assert evoforge.Message is not None


def test_evoforge_submodule_alias_resolves():
    from evoforge.core.agent_config import AgentConfig  # noqa: F401


def test_env_namespace_connect_lifecycle():
    sdk = foundry.init(task_spec="P0 env wiring test")

    class DummyEnv:
        def reset(self, seed):  # noqa: ANN001
            return {}

        def step(self, action):  # noqa: ANN001
            return {}

        def get_state(self):
            return {}

        def check_goal(self, gold):  # noqa: ANN001
            return {}

        def check_milestone(self, milestone):  # noqa: ANN001
            return True

        def inject_failure(self, config):  # noqa: ANN001
            return None

        def snapshot(self):
            return {}

        def restore(self, snapshot):  # noqa: ANN001
            return None

        def close(self):
            return None

    connector = DummyEnv()
    connected = sdk.env.connect(connector)
    assert connected is connector
    assert sdk.env.is_connected() is True
    assert sdk.env.connector is connector

    sdk.env.close()
    assert sdk.env.is_connected() is False
    assert sdk.env.connector is None


def test_env_namespace_rejects_invalid_connector():
    sdk = foundry.init(task_spec="P0 invalid env wiring test")

    class InvalidEnv:
        def close(self):
            return None

    with pytest.raises(TypeError):
        sdk.env.connect(InvalidEnv())


def test_report_command_generates_html(tmp_path: Path):
    runner = CliRunner()

    work_dir = tmp_path / "workspace"
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / ".foundry").mkdir(parents=True, exist_ok=True)
    output = work_dir / "report.html"

    prev_cwd = Path.cwd()
    os.chdir(work_dir)
    try:
        result = runner.invoke(cli, ["report", "--output", str(output)])
    finally:
        os.chdir(prev_cwd)

    assert result.exit_code == 0
    assert output.exists()
