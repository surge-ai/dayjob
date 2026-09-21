"""Tests for the ATIF trajectory builder.

``events_to_steps`` is exercised against real (pinned) SDK event objects, so
these tests also pin the runner's assumptions about the SDK event schema.
"""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from openhands.sdk.event import ActionEvent, MessageEvent, ObservationEvent  # noqa: E402
from openhands.sdk.llm import Message, TextContent  # noqa: E402
from openhands.sdk.llm.message import MessageToolCall  # noqa: E402
from openhands.sdk.tool.schema import Observation  # noqa: E402

from agent_harness.atif import build_trajectory, events_to_steps  # noqa: E402


class _Obs(Observation):
    """Minimal concrete Observation for constructing ObservationEvents."""


def _message_event(source, text):
    role = "user" if source == "user" else "assistant"
    return MessageEvent(
        source=source,
        llm_message=Message(role=role, content=[TextContent(text=text)]),
    )


def _action_event(arguments='{"command": "ls"}', call_id="call-1", **kwargs):
    return ActionEvent(
        thought=[TextContent(text="I'll list files")],
        tool_name="terminal",
        tool_call_id=call_id,
        tool_call=MessageToolCall(
            id=call_id, name="terminal", arguments=arguments, origin="completion"
        ),
        llm_response_id="resp-1",
        **kwargs,
    )


def _observation_event(action, text="file.txt", call_id="call-1"):
    return ObservationEvent(
        observation=_Obs(content=[TextContent(text=text)]),
        action_id=action.id,
        tool_name="terminal",
        tool_call_id=call_id,
    )


class TestEventsToSteps:
    def test_user_and_agent_messages(self):
        steps = events_to_steps(
            [_message_event("user", "do it"), _message_event("agent", "done")],
            model_name="openai/gpt-5",
        )
        assert [(s["source"], s["message"]) for s in steps] == [
            ("user", "do it"),
            ("agent", "done"),
        ]
        # Agent steps carry the model name; user steps don't.
        assert "model_name" not in steps[0]
        assert steps[1]["model_name"] == "openai/gpt-5"

    def test_action_event_carries_thought_and_arguments(self):
        [step] = events_to_steps([_action_event()], model_name="m")
        # The agent's inter-tool commentary is preserved as the message.
        assert step["message"] == "I'll list files"
        assert step["tool_calls"] == [
            {
                "tool_call_id": "call-1",
                "function_name": "terminal",
                "arguments": {"command": "ls"},
            }
        ]

    def test_unparseable_arguments_kept_raw(self):
        [step] = events_to_steps([_action_event(arguments="not json")], model_name="m")
        assert step["tool_calls"][0]["arguments"] == {"raw": "not json"}

    def test_observation_attaches_to_its_action_step(self):
        action = _action_event()
        steps = events_to_steps(
            [action, _observation_event(action)], model_name="m"
        )
        assert len(steps) == 1
        assert steps[0]["observation"]["results"] == [
            {"source_call_id": "call-1", "content": "file.txt"}
        ]

    def test_orphan_observation_is_dropped(self):
        action = _action_event()
        steps = events_to_steps([_observation_event(action)], model_name="m")
        assert steps == []

    def test_parallel_tool_calls_match_observations_by_id(self):
        # One LLM response with two tool calls arrives as consecutive
        # ActionEvents; observations must land on the step holding their
        # call, not on whichever step happens to be last.
        first = _action_event(call_id="call-1")
        second = _action_event(call_id="call-2")
        steps = events_to_steps(
            [
                first,
                second,
                _observation_event(first, text="from-1", call_id="call-1"),
                _observation_event(second, text="from-2", call_id="call-2"),
            ],
            model_name="m",
        )
        assert steps[0]["observation"]["results"] == [
            {"source_call_id": "call-1", "content": "from-1"}
        ]
        assert steps[1]["observation"]["results"] == [
            {"source_call_id": "call-2", "content": "from-2"}
        ]


class TestBuildTrajectory:
    def test_document_shape(self):
        events = [
            _message_event("user", "do it"),
            _action_event(),
            _message_event("agent", "done"),
        ]
        doc = build_trajectory(
            events_to_steps(events, model_name="openai/gpt-5"),
            llm_metrics={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cached_tokens": 5,
                "cost_usd": 0.7,
            },
            agent_name="openhands_sdk",
            agent_version="1.2.3",
            system_prompt="be helpful",
            extra={"final_output": "answer", "stopped_reason": "end_turn"},
        )

        assert doc["schema_version"].startswith("ATIF-")
        # Required by the stamped schema version (only v1.7+ makes it optional).
        assert doc["session_id"]
        assert doc["agent"] == {
            "name": "openhands_sdk",
            "version": "1.2.3",
            "model_name": None,
            "tool_definitions": None,
        }
        # System prompt is the first step; IDs are sequential from 1.
        assert [s["step_id"] for s in doc["steps"]] == [1, 2, 3, 4]
        assert doc["steps"][0]["source"] == "system"
        assert doc["steps"][0]["message"] == "be helpful"
        assert doc["steps"][2]["tool_calls"][0]["function_name"] == "terminal"
        assert doc["final_metrics"] == {
            "total_prompt_tokens": 100,
            "total_completion_tokens": 20,
            "total_cached_tokens": 5,
            "total_cost_usd": 0.7,
        }
        assert doc["extra"]["final_output"] == "answer"

    def test_empty_run_produces_valid_document(self):
        doc = build_trajectory(
            [], llm_metrics={}, agent_name="openhands_sdk", agent_version="unknown"
        )
        assert doc["steps"] == []
        assert doc["session_id"]
        # Unknown metrics are omitted, not reported as zeros.
        assert doc["final_metrics"] == {}
        assert "extra" not in doc
