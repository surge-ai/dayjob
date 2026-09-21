"""Runner contracts that exercise the pinned OpenHands SDK (offline).

These import the SDK, so they pin the runner's assumptions about its API:
the MCP config schema, the conversation status values behind stopped_reason,
and persistence of clipped tool output.
"""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from openhands.sdk import ConversationExecutionStatus  # noqa: E402
from openhands.sdk.event.conversation_error import ConversationErrorEvent  # noqa: E402
from openhands.sdk.llm.message import Message  # noqa: E402
from openhands.sdk.mcp.config import MCPServer  # noqa: E402

from agent_harness import LLM_CREDENTIAL_ENV_VARS  # noqa: E402
from agent_harness.openhands_patches import (  # noqa: E402
    hide_credentials_from_tools,
    persist_large_tool_outputs,
)
from agent_harness.runner import (  # noqa: E402
    _mcp_config,
    _stopped_reason,
)


def test_default_tools_can_be_disabled():
    # The runner passes include_default_tools=[] to drop the SDK's finish
    # and think tools. Pin both sides: the SDK still injects exactly those
    # two by default (so we know what we're opting out of), and an empty
    # list is honored.
    from openhands.sdk import LLM, Agent

    llm = LLM(usage_id="test", model="openai/gpt-5")
    assert Agent(llm=llm, tools=[], mcp_config={}).include_default_tools == [
        "FinishTool",
        "ThinkTool",
    ]
    agent = Agent(llm=llm, tools=[], mcp_config={}, include_default_tools=[])
    assert agent.include_default_tools == []


def test_mcp_config_validates_against_sdk_schema():
    # Regression: the pre-1.43 {"mcpServers": {...}} wrapper shape fails this
    # with extra_forbidden, killing every MCP task at agent construction.
    cfg = _mcp_config("http://localhost:8000/mcp")
    for name, server in cfg.items():
        validated = MCPServer.model_validate(server)
        assert validated.url == "http://localhost:8000/mcp"
        assert validated.timeout == 300


class TestStoppedReason:
    def _status(self, status):
        return status.value

    def test_run_exception_is_error_even_after_max_turns(self):
        status = self._status(ConversationExecutionStatus.ERROR)
        assert _stopped_reason(True, status, hit_max_turns=True) == "error"

    def test_error_status_without_max_turns_is_error(self):
        status = self._status(ConversationExecutionStatus.ERROR)
        assert _stopped_reason(False, status, hit_max_turns=False) == "error"

    def test_max_turns_is_not_a_trial_error(self):
        status = self._status(ConversationExecutionStatus.ERROR)
        assert _stopped_reason(False, status, hit_max_turns=True) == "max_turns"

    def test_stuck(self):
        status = self._status(ConversationExecutionStatus.STUCK)
        assert _stopped_reason(False, status, hit_max_turns=False) == "stuck"

    def test_finished_is_end_turn(self):
        status = self._status(ConversationExecutionStatus.FINISHED)
        assert _stopped_reason(False, status, hit_max_turns=False) == "end_turn"

    def test_unknown_status_is_not_assumed_to_be_max_turns(self):
        assert _stopped_reason(False, None, hit_max_turns=False) == "error"


def test_sdk_iteration_limit_emits_expected_code(monkeypatch, tmp_path):
    from openhands.sdk import LLM, Agent, Conversation

    # Exercise the SDK limit without LLM requests or tool execution.
    monkeypatch.setattr(Agent, "step", lambda *args, **kwargs: None)
    events = []
    conversation = Conversation(
        agent=Agent(
            llm=LLM(usage_id="test", model="openai/gpt-5"),
            tools=[], include_default_tools=[],
        ),
        workspace=str(tmp_path),
        persistence_dir=str(tmp_path / "state"),
        callbacks=[events.append],
        max_iteration_per_run=1,
    )
    conversation.send_message("test")
    conversation.run()
    errors = [event for event in events if isinstance(event, ConversationErrorEvent)]
    assert conversation.state.execution_status == ConversationExecutionStatus.ERROR
    assert errors[-1].code == "MaxIterationsReached"


def test_large_tool_output_is_saved_before_clipping(tmp_path):
    import openhands.sdk.llm.message as message_mod

    original = Message._maybe_truncate_tool_text
    try:
        persist_large_tool_outputs(str(tmp_path))
        text = "x" * (message_mod.DEFAULT_TEXT_CONTENT_LIMIT + 1)
        message = Message(role="tool", content=[], tool_call_id="call_fixture")
        clipped = message._maybe_truncate_tool_text(text)
    finally:
        Message._maybe_truncate_tool_text = original

    files = list(tmp_path.iterdir())
    assert len(clipped) == message_mod.DEFAULT_TEXT_CONTENT_LIMIT
    assert "complete output has been saved" in clipped
    assert len(files) == 1
    assert files[0].read_text() == text


def test_forwarded_credentials_are_hidden_from_tool_processes(monkeypatch):
    import openhands.sdk.utils.command as command_mod
    from openhands.tools.terminal.env import build_terminal_env

    for name in LLM_CREDENTIAL_ENV_VARS:
        monkeypatch.setenv(name, f"secret-{name}")
    monkeypatch.setenv("TASK_SETTING", "keep")
    monkeypatch.setattr(command_mod, "_SENSITIVE_ENV_VARS", command_mod._SENSITIVE_ENV_VARS)
    assert build_terminal_env()["OPENAI_API_KEY"] == "secret-OPENAI_API_KEY"

    hide_credentials_from_tools()
    hide_credentials_from_tools()

    tool_env = build_terminal_env()
    for name in LLM_CREDENTIAL_ENV_VARS:
        assert name not in tool_env
        assert os.environ[name] == f"secret-{name}"
    assert tool_env["TASK_SETTING"] == "keep"
    assert "SESSION_API_KEY" in command_mod._SENSITIVE_ENV_VARS
