"""Run DAYJOB trials with the OpenHands SDK.

Every run includes local terminal, file-editor, and task-tracker tools.
Setting ``mcpUrl`` adds tools from an MCP server. Local tools operate in
``workspaceDir``. The SDK's default finish and think tools are disabled;
the agent ends its turn with a plain message.

Contract:
- argv[1] = config JSON path; argv[2] = trajectory JSON output path (a
  dedicated file — stdout carries the SDK's human-readable transcript).
- The trajectory is an ATIF document (harbor's Agent Trajectory Interchange
  Format): the full step record lives in ``steps``, and the harness's run
  verdict (``TrajectoryExtras``) lives under ATIF's ``extra`` field.
- Exits 0 even when the loop errored; the error is recorded in the
  summary's ``stopped_reason``/``error_message`` for the host to act on.
"""

from __future__ import annotations

import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any, Literal, Required, TypedDict, cast

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    ConversationExecutionStatus,
    Event,
    LLMSummarizingCondenser,
    MessageEvent,
    Tool,
)
from openhands.sdk.event import AgentErrorEvent
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.llm.message import content_to_str

# Importing a tool module registers it, so Tool(name=...) can resolve it.
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from agent_harness.atif import build_trajectory, events_to_steps
from agent_harness.openhands_patches import (
    configure_condenser_transport,
    hide_credentials_from_tools,
    install_model_capabilities,
    install_responses_replay,
    persist_large_tool_outputs,
)

AGENT_ID = "openhands_sdk"
MCP_SERVER_NAME = "dayjob"
MCP_TOOL_TIMEOUT = 300
LLM_TIMEOUT = 600
WORKSPACE_DIR = "/tmp/openhands_workspace"
STATE_DIR = "/tmp/openhands_state"
DEFAULT_REASONING_SUMMARY = "auto"


class RunConfig(TypedDict, total=False):
    """Config JSON consumed by ``run()`` (argv[1] of the CLI contract)."""

    model: Required[str]
    """litellm model string, e.g. ``openai/gpt-5``."""

    instruction: Required[str]
    """Task prompt, sent as the first user message."""

    maxTurns: Required[int]
    """Hard cap on agent turns, where a turn is one LLM call and the tool
    calls it issues (SDK ``max_iteration_per_run``)."""

    systemPrompt: str
    """Replaces the SDK's default system prompt verbatim."""

    mcpUrl: str
    """Optional MCP server URL. Adds server tools to the local toolset."""

    workspaceDir: str
    """Agent working directory. Defaults to ``WORKSPACE_DIR``."""

    logDir: str
    """Parent directory for completion logs when logging is enabled.
    Logs are written to its ``completions/`` subdirectory.
    Defaults to the ``WORKSPACE_DIR`` constant."""

    llmKwargs: dict[str, Any]
    """Forwarded to the SDK ``LLM`` constructor, except the intercepted
    keys ``reasoning_effort``, ``reasoning_summary``, ``thought_display``,
    and ``log_completions`` (see ``_resolve_llm_kwargs``)."""


StoppedReason = Literal["end_turn", "max_turns", "stuck", "error"]


class TrajectoryExtras(TypedDict):
    """Harness verdict embedded at the ATIF trajectory's ``extra`` field:
    only what cannot be derived from the rest of the document. (Token/cost
    totals live in ``final_metrics``; model and agent identity in ``agent``;
    tool calls are countable from ``steps``.) Every key is always present;
    hosts should branch on ``stopped_reason`` rather than the exit code
    (the process exits 0 even when the run errored)."""

    final_output: str
    """The agent's final answer: the last agent message. Empty if the
    agent never responded."""

    stopped_reason: StoppedReason
    error_message: str | None

    n_agent_errors: int
    """Agent error events, which the step record omits."""

    reasoning_tokens: int | float | None
    """Summed across agent and condenser LLMs; ``final_metrics`` has no
    slot for reasoning tokens. ``None`` means unknown."""


def _validate_config(config: dict[str, Any]) -> RunConfig:
    """Check the required ``RunConfig`` keys, failing with a clear message
    instead of a bare ``KeyError`` mid-run."""
    missing = [key for key in ("model", "instruction", "maxTurns") if key not in config]
    if missing:
        raise ValueError(f"config is missing required key(s): {', '.join(missing)}")
    return cast(RunConfig, config)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _resolve_llm_kwargs(
    config: RunConfig,
) -> tuple[dict[str, Any], str | None, str | None]:
    """Split LLM kwargs into ``(constructor_kwargs, reasoning_effort,
    reasoning_summary)``.

    ``reasoning_effort`` is applied post-construction: the SDK constructor
    rejects values like ``max``. ``reasoning_summary`` defaults to ``auto``
    so Responses requests surface reasoning summaries; pass
    ``reasoning_summary=none`` to disable.
    """
    llm_kwargs = dict(config.get("llmKwargs") or {})

    effort = llm_kwargs.pop("reasoning_effort", None)
    if effort is not None:
        effort = str(effort).strip().lower() or None

    summary = llm_kwargs.pop("reasoning_summary", DEFAULT_REASONING_SUMMARY)
    if summary is not None:
        summary = str(summary).strip().lower() or None
        if summary in ("none", "off"):
            summary = None

    # thought_display (MAI endpoints) is a request-body param, not an LLM
    # constructor kwarg; route it through litellm_extra_body.
    thought_display = llm_kwargs.pop("thought_display", None)
    if thought_display:
        llm_kwargs["litellm_extra_body"] = {
            **llm_kwargs.get("litellm_extra_body", {}),
            "thought_display": str(thought_display),
        }

    return llm_kwargs, effort, summary


def _usage(*llms: Any) -> dict[str, int | float | None]:
    """Sum token/cost totals across LLMs (agent + condenser), tolerant of
    metrics-API drift: fields stay ``None`` rather than crashing the trial."""
    input_tokens = output_tokens = reasoning_tokens = cache_tokens = None
    cost: float | None = None

    def _accumulate(current: int | float | None, value: Any) -> int | float | None:
        if not isinstance(value, (int, float)):
            return current
        return value if current is None else current + value

    for llm in llms:
        metrics = getattr(llm, "metrics", None)
        if metrics is None:
            continue
        cost = _accumulate(cost, getattr(metrics, "accumulated_cost", None))
        token_usage = getattr(metrics, "accumulated_token_usage", None)
        if token_usage is None:
            continue
        input_tokens = _accumulate(input_tokens, getattr(token_usage, "prompt_tokens", None))
        output_tokens = _accumulate(output_tokens, getattr(token_usage, "completion_tokens", None))
        reasoning_tokens = _accumulate(
            reasoning_tokens, getattr(token_usage, "reasoning_tokens", None)
        )
        cache_tokens = _accumulate(cache_tokens, getattr(token_usage, "cache_read_tokens", None))

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_tokens": cache_tokens,
        # accumulated_cost is 0.0 when litellm has no pricing for the model
        # (e.g. an aliased model behind a proxy); treat that as "unknown" so
        # downstream cost reporting can fall back to provider dashboards.
        "cost_usd": cost if cost else None,
    }


def _mcp_config(mcp_url: str | None) -> dict[str, Any]:
    """Agent ``mcp_config`` for the task's proxy, empty when proxy-less.

    SDK >= 1.43 takes server-name -> MCPServer directly; the old
    ``{"mcpServers": {...}}`` wrapper is rejected with extra_forbidden.
    """
    if not mcp_url:
        return {}
    return {
        MCP_SERVER_NAME: {
            "url": mcp_url,
            "timeout": MCP_TOOL_TIMEOUT,
        }
    }


def _stopped_reason(saw_error: bool, status_val: Any, hit_max_turns: bool) -> StoppedReason:
    """Map the conversation's end state to the trajectory's stopped_reason."""
    if saw_error:
        # Raised exceptions take precedence over turn exhaustion.
        return "error"
    if status_val == ConversationExecutionStatus.ERROR.value and not hit_max_turns:
        return "error"
    if status_val == ConversationExecutionStatus.STUCK.value:
        return "stuck"
    if status_val == ConversationExecutionStatus.FINISHED.value:
        return "end_turn"
    return "max_turns" if hit_max_turns else "error"


def _agent_version() -> str:
    try:
        return _pkg_version("agent-harness")
    except PackageNotFoundError:
        return "unknown"


def _atif_document(
    extras: TrajectoryExtras,
    model: str | None,
    usage: dict[str, int | float | None] | None = None,
    events: Any = (),
    system_prompt: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the ATIF trajectory written to argv[2]: the step record from
    the SDK events plus the harness verdict under ``extra``."""
    usage = usage or {}
    return build_trajectory(
        events_to_steps(events, model_name=model or ""),
        {
            # None (metrics API unavailable) passes through and is omitted
            # from final_metrics, keeping "unknown" distinct from a real 0.
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "cached_tokens": usage.get("cache_tokens"),
            "cost_usd": usage.get("cost_usd"),
        },
        agent_name=AGENT_ID,
        agent_version=_agent_version(),
        model_name=model,
        system_prompt=system_prompt,
        tool_definitions=tool_definitions,
        extra=dict(extras),
    )


def run(raw_config: dict[str, Any]) -> dict[str, Any]:
    config = _validate_config(raw_config)
    install_responses_replay()
    install_model_capabilities()
    hide_credentials_from_tools()

    litellm_model = config["model"]
    llm_kwargs, reasoning_effort, reasoning_summary = _resolve_llm_kwargs(config)
    # Behind a proxy, config["model"] is an alias (litellm_proxy/...); the real
    # model id is in model_canonical_name. Use it for the provider checks below,
    # matching what the SDK uses for capability resolution.
    routing_model = str(llm_kwargs.get("model_canonical_name") or litellm_model)

    # Opt-in per-call request logging (--ak log_completions=true), written to
    # the trial's log dir so it's downloaded with the trial.
    if llm_kwargs.pop("log_completions", False):
        log_root = config.get("logDir") or WORKSPACE_DIR
        llm_kwargs["log_completions"] = True
        llm_kwargs["log_completions_folder"] = os.path.join(log_root, "completions")
        _log(f"completion logging enabled -> {llm_kwargs['log_completions_folder']}")

    def _make_llm(usage_id: str) -> Any:
        return LLM(
            usage_id=usage_id,
            model=litellm_model,
            timeout=LLM_TIMEOUT,
            **llm_kwargs,
        )

    llm = _make_llm("agent")
    condenser_llm = _make_llm("condenser")

    reasoning: dict[str, str] = {}
    for _llm in (llm, condenser_llm):
        if reasoning_effort == "none":
            # Drop the param entirely (the SDK defaults reasoning models to
            # "high"); some proxies reject any reasoning_effort alongside
            # tools.
            _llm.reasoning_effort = None
        elif reasoning_effort is not None:
            # Bare prefixes (claude-, gemini-) cover ids passed without a
            # provider prefix, which litellm resolves via its registry.
            if routing_model.startswith(("anthropic/", "claude-", "gemini/", "gemini-")):
                # litellm translates the native param per provider (Gemini:
                # thinkingConfig.thinkingLevel). The extra_body form below is
                # a Responses-API construct that Gemini's API rejects with
                # 400 'Unknown name "reasoning"'.
                _llm.reasoning_effort = reasoning_effort
            else:
                reasoning["effort"] = reasoning_effort
    # Summaries are a Responses-API feature; Chat requests reject them.
    if reasoning_summary is not None and llm.uses_responses_api():
        reasoning["summary"] = reasoning_summary
    if reasoning:
        for _llm in (llm, condenser_llm):
            _llm.litellm_extra_body = {
                **_llm.litellm_extra_body,
                "reasoning": reasoning,
            }

    configure_condenser_transport(condenser_llm)

    mcp_url = config.get("mcpUrl")
    workspace_dir = config.get("workspaceDir") or WORKSPACE_DIR
    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]
    mcp_config = _mcp_config(mcp_url)

    agent_kwargs = {}
    if config.get("systemPrompt"):
        agent_kwargs["system_prompt"] = config["systemPrompt"]
    agent = Agent(
        llm=llm,
        tools=tools,
        mcp_config=mcp_config,
        # Disable the default finish and think tools. Final answers arrive
        # as ordinary messages, which the callback captures below.
        include_default_tools=[],
        condenser=LLMSummarizingCondenser(llm=condenser_llm),
        **agent_kwargs,
    )

    hit_max_turns = False
    n_agent_errors = 0
    final_output = ""
    error_message: str | None = None

    def callback(event: Event) -> None:
        nonlocal hit_max_turns, n_agent_errors, final_output, error_message
        if isinstance(event, ConversationErrorEvent):
            hit_max_turns = event.code == "MaxIterationsReached"
        elif isinstance(event, AgentErrorEvent):
            n_agent_errors += 1
            error_message = getattr(event, "error", None) or str(event)
        elif isinstance(event, MessageEvent) and event.source == "agent":
            text = "".join(content_to_str(event.to_llm_message().content))
            if text:
                final_output = text

    conversation = Conversation(
        agent=agent,
        callbacks=[callback],
        workspace=workspace_dir,
        persistence_dir=STATE_DIR,
        max_iteration_per_run=int(config["maxTurns"]),
    )
    observation_dir = conversation.state.env_observation_persistence_dir
    assert observation_dir is not None
    persist_large_tool_outputs(observation_dir)

    saw_error = False
    try:
        conversation.send_message(config["instruction"])
        conversation.run()
    except Exception as e:  # provider/transport/loop failure
        saw_error = True
        error_message = error_message or str(e)
        _log(f"OpenHands conversation.run() raised: {e}")

    status = getattr(conversation.state, "execution_status", None)
    status_val = getattr(status, "value", status)
    stopped_reason = _stopped_reason(saw_error, status_val, hit_max_turns)

    usage = _usage(llm, condenser_llm)

    extras = TrajectoryExtras(
        final_output=final_output,
        stopped_reason=stopped_reason,
        error_message=error_message,
        n_agent_errors=n_agent_errors,
        reasoning_tokens=usage["reasoning_tokens"],
    )

    # Step record for the ATIF document; the summary is authoritative, so
    # extraction failures degrade the record rather than the trial.
    system_prompt: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    try:
        system_prompt = agent.static_system_message
    except Exception as e:
        _log(f"could not extract system prompt: {e}")
    try:
        tool_definitions = [
            tool.to_openai_tool() for tool in agent.tools_map.values()
        ]
    except Exception as e:
        _log(f"could not extract tool definitions: {e}")
    events = list(conversation.state.events)

    return _atif_document(
        extras, litellm_model, usage, events, system_prompt, tool_definitions
    )


def main() -> None:
    if len(sys.argv) != 3:
        _log("usage: runner.py <config.json> <output.json>")
        sys.exit(2)

    config_path, output_path = sys.argv[1], sys.argv[2]
    with open(config_path) as f:
        config = json.load(f)
    try:
        result = run(config)
    except Exception as e:
        # Hard failure before/around the loop — still emit a trajectory so
        # the host sees an errored trial rather than a crash.
        result = _atif_document(
            TrajectoryExtras(
                final_output="",
                stopped_reason="error",
                error_message=str(e),
                n_agent_errors=0,
                reasoning_tokens=None,
            ),
            config.get("model"),
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    extras = result.get("extra") or {}
    _log(
        f"wrote trajectory to {output_path} "
        f"(stopped_reason={extras.get('stopped_reason')}, "
        f"n_steps={len(result.get('steps') or [])})"
    )


if __name__ == "__main__":
    main()
