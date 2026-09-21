import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from harbor.agents.installed.base import BaseInstalledAgent

from harbor_agents import HarborInstalledAgent
from harbor_agents.agents import (
    _apply_trajectory,
    _pack_runner_project,
    _pop_llm_kwargs,
    _run_config,
)


def test_installed_agent_subclasses_harbor_installed_agent():
    assert issubclass(HarborInstalledAgent, BaseInstalledAgent)


def test_pack_runner_project_matches_staged_layout(tmp_path: Path):
    # The self-install tarball must extract into the same layout that
    # stage_openhands_runner bakes into images: pyproject + uv.lock + src.
    tarball = tmp_path / "openhands-runner.tgz"
    _pack_runner_project(tarball)

    with tarfile.open(tarball) as tar:
        names = tar.getnames()
    assert "pyproject.toml" in names
    assert "uv.lock" in names
    assert "src/agent_harness/runner.py" in names
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]


def test_run_config_has_no_bash_switch():
    config = _run_config(
        instruction="test",
        system_prompt=None,
        model="openai/gpt-5",
        mcp_url="http://localhost:8000/mcp",
        workspace_dir="/workdir",
        llm_kwargs={},
        log_dir="/logs/agent",
    )
    assert "bashTool" not in config
    assert config["maxTurns"] == 1000
    assert "maxToolCalls" not in config


def test_openhands_llm_kwargs_are_forwarded():
    kwargs = {
        "api_mode": "responses",
        "max_output_tokens": 64000,
        "reasoning_effort": "xhigh",
        "reasoning_summary": "auto",
    }

    assert _pop_llm_kwargs(kwargs) == {
        "api_mode": "responses",
        "max_output_tokens": 64000,
        "reasoning_effort": "xhigh",
        "reasoning_summary": "auto",
    }


def test_apply_trajectory_reads_atif_document():
    # Verdict from ``extra``, tokens/cost from ``final_metrics``, identity
    # from ``agent``, tool calls counted from ``steps``.
    context = SimpleNamespace()
    _apply_trajectory(
        {
            "schema_version": "ATIF-v1.5",
            "agent": {"name": "openhands_sdk", "model_name": "openai/gpt-5"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "hi"},
                {"step_id": 2, "source": "agent", "tool_calls": [{"tool_call_id": "c1"}]},
                {"step_id": 3, "source": "agent", "message": "answer"},
            ],
            "final_metrics": {
                "total_prompt_tokens": 100,
                "total_completion_tokens": 50,
                "total_cached_tokens": 10,
                "total_cost_usd": 0.4,
            },
            "extra": {
                "stopped_reason": "end_turn",
                "final_output": "answer",
                "reasoning_tokens": 30,
            },
        },
        context,
    )
    assert context.n_input_tokens == 100
    assert context.n_output_tokens == 50
    assert context.n_cache_tokens == 10
    assert context.cost_usd == 0.4
    assert context.metadata["agent_id"] == "openhands_sdk"
    assert context.metadata["model"] == "openai/gpt-5"
    assert context.metadata["n_tool_calls"] == 1
    assert context.metadata["reasoning_tokens"] == 30
    assert context.metadata["stopped_reason"] == "end_turn"
    assert context.metadata["final_output_chars"] == len("answer")


def test_apply_trajectory_zero_cost_is_unknown():
    # The runner writes 0.0 when litellm has no pricing for the model.
    context = SimpleNamespace()
    _apply_trajectory(
        {"final_metrics": {"total_cost_usd": 0.0}, "extra": {}}, context
    )
    assert context.cost_usd is None


def test_apply_trajectory_rejects_pre_atif_format():
    # Old runners wrote the summary as the whole document; that format is
    # no longer supported and should fail loudly, not read Nones.
    with pytest.raises(RuntimeError, match="no run summary under 'extra'"):
        _apply_trajectory(
            {"input_tokens": 100, "stopped_reason": "end_turn"},
            SimpleNamespace(),
        )
