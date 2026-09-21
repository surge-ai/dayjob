"""Tests for the runner's pure helpers (no SDK objects exercised)."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_harness.runner import (
    DEFAULT_REASONING_SUMMARY,
    _mcp_config,
    _resolve_llm_kwargs,
    _usage,
    main,
)


class TestResolveLlmKwargs:
    def test_empty_config(self):
        assert _resolve_llm_kwargs({}) == ({}, None, DEFAULT_REASONING_SUMMARY)

    def test_passes_through_plain_kwargs(self):
        kwargs, effort, _ = _resolve_llm_kwargs({"llmKwargs": {"temperature": 0}})
        assert kwargs == {"temperature": 0}
        assert effort is None

    def test_extracts_and_normalizes_reasoning_effort(self):
        kwargs, effort, _ = _resolve_llm_kwargs(
            {"llmKwargs": {"reasoning_effort": "  MAX  ", "temperature": 0}}
        )
        assert effort == "max"
        assert kwargs == {"temperature": 0}

    def test_blank_reasoning_effort_becomes_none(self):
        _, effort, _ = _resolve_llm_kwargs({"llmKwargs": {"reasoning_effort": "   "}})
        assert effort is None

    def test_reasoning_summary_defaults_to_auto_and_can_be_disabled(self):
        _, _, summary = _resolve_llm_kwargs({})
        assert summary == DEFAULT_REASONING_SUMMARY
        _, _, disabled = _resolve_llm_kwargs(
            {"llmKwargs": {"reasoning_summary": "none"}}
        )
        assert disabled is None

    def test_routes_thought_display_through_litellm_extra_body(self):
        kwargs, _, _ = _resolve_llm_kwargs({"llmKwargs": {"thought_display": "full"}})
        assert kwargs == {"litellm_extra_body": {"thought_display": "full"}}

    def test_thought_display_merges_with_existing_extra_body(self):
        kwargs, _, _ = _resolve_llm_kwargs(
            {
                "llmKwargs": {
                    "thought_display": "full",
                    "litellm_extra_body": {"other": 1},
                }
            }
        )
        assert kwargs == {
            "litellm_extra_body": {"other": 1, "thought_display": "full"}
        }


def _llm(cost=None, prompt=None, completion=None, reasoning=None, cache=None, metrics=True):
    if not metrics:
        return SimpleNamespace(metrics=None)
    token_usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        cache_read_tokens=cache,
    )
    return SimpleNamespace(
        metrics=SimpleNamespace(accumulated_cost=cost, accumulated_token_usage=token_usage)
    )


class TestUsage:
    def test_no_llms_yields_all_none(self):
        assert _usage() == {
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "cache_tokens": None,
            "cost_usd": None,
        }

    def test_sums_across_agent_and_condenser(self):
        # reasoning_tokens ride alongside output_tokens (the SDK reports them
        # separately) — summing must not double-count them into output.
        agent = _llm(cost=1.5, prompt=100, completion=20, reasoning=30, cache=10)
        condenser = _llm(cost=0.5, prompt=50, completion=5, reasoning=4, cache=0)
        assert _usage(agent, condenser) == {
            "input_tokens": 150,
            "output_tokens": 25,
            "reasoning_tokens": 34,
            "cache_tokens": 10,
            "cost_usd": 2.0,
        }

    def test_tolerates_missing_metrics(self):
        usage = _usage(_llm(metrics=False), _llm(cost=1.0, prompt=10))
        assert usage["input_tokens"] == 10
        assert usage["cost_usd"] == 1.0

    def test_zero_cost_reports_unknown(self):
        # litellm reports 0.0 when it has no pricing for the model; the
        # trajectory reports None so downstream falls back to dashboards.
        assert _usage(_llm(cost=0.0, prompt=10))["cost_usd"] is None

    def test_non_numeric_values_are_ignored(self):
        broken = SimpleNamespace(
            metrics=SimpleNamespace(
                accumulated_cost="n/a",
                accumulated_token_usage=SimpleNamespace(
                    prompt_tokens="many",
                    completion_tokens=None,
                    reasoning_tokens=None,
                    cache_read_tokens=None,
                ),
            )
        )
        assert _usage(broken) == {
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "cache_tokens": None,
            "cost_usd": None,
        }


class TestMcpConfig:
    def test_no_proxy_yields_empty_config(self):
        assert _mcp_config(None) == {}
        assert _mcp_config("") == {}

    def test_server_is_keyed_by_name_without_wrapper(self):
        cfg = _mcp_config("http://localhost:8000/mcp")
        # SDK >= 1.43 rejects the old {"mcpServers": {...}} wrapper shape.
        assert "mcpServers" not in cfg
        assert cfg["dayjob"]["url"] == "http://localhost:8000/mcp"
        assert cfg["dayjob"]["timeout"] > 0


class TestMain:
    def _config(self, tmp_path, **overrides):
        config = {"model": "openai/gpt-5", "instruction": "do it", **overrides}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        return path

    def test_wrong_argv_exits_2(self):
        with patch("sys.argv", ["runner.py"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 2

    def test_writes_run_result_to_output_path(self, tmp_path):
        config_path = self._config(tmp_path)
        out_path = tmp_path / "nested" / "trajectory.json"
        result = {"agent_id": "openhands_sdk", "stopped_reason": "end_turn"}
        with (
            patch("sys.argv", ["runner.py", str(config_path), str(out_path)]),
            patch("agent_harness.runner.run", return_value=result) as run,
        ):
            main()
        # Creates missing parent directories and round-trips the result.
        assert json.loads(out_path.read_text()) == result
        assert run.call_args.args[0]["instruction"] == "do it"

    def test_run_crash_still_writes_error_trajectory(self, tmp_path):
        config_path = self._config(tmp_path)
        out_path = tmp_path / "trajectory.json"
        with (
            patch("sys.argv", ["runner.py", str(config_path), str(out_path)]),
            patch(
                "agent_harness.runner.run",
                side_effect=RuntimeError("proxy exploded"),
            ),
        ):
            main()  # must not raise: the host reads the trajectory instead
        result = json.loads(out_path.read_text())
        # The error document is still an ATIF trajectory with the verdict
        # under ``extra`` and the model recorded in the agent block.
        assert result["schema_version"].startswith("ATIF-")
        assert result["steps"] == []
        assert result["agent"]["model_name"] == "openai/gpt-5"
        extras = result["extra"]
        assert extras["stopped_reason"] == "error"
        assert "proxy exploded" in extras["error_message"]
        assert extras["final_output"] == ""
