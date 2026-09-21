"""Offline contracts for Responses history on the pinned OpenHands SDK.

These exercise the runtime patch in ``openhands_patches``: the
complete ordered Responses output must survive message conversion, event JSON
persistence and reload, multi-action recombination, and re-serialization for
the next request.
"""

from __future__ import annotations

import os
import socket
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
_home = tempfile.TemporaryDirectory(prefix="agent-harness-test-home-")
_original_home = os.environ.get("HOME")
os.environ["HOME"] = _home.name
_network_patchers = [
    patch.object(
        socket.socket,
        "connect",
        side_effect=AssertionError("Responses tests must remain offline"),
    ),
    patch.object(
        socket,
        "create_connection",
        side_effect=AssertionError("Responses tests must remain offline"),
    ),
]


def tearDownModule() -> None:
    for patcher in reversed(_network_patchers):
        patcher.stop()
    if _original_home is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = _original_home
    _home.cleanup()


for _patcher in _network_patchers:
    _patcher.start()

from agent_harness.openhands_patches import (  # noqa: E402
    install_responses_replay,
)

install_responses_replay()
install_responses_replay()  # idempotent

from openhands.sdk.event import ActionEvent, LLMConvertibleEvent  # noqa: E402
from openhands.sdk.llm import Message  # noqa: E402
from openhands.sdk.llm.utils.responses_serialization import (  # noqa: E402
    _assistant_to_responses_items,
)


def reasoning(index: int) -> dict:
    return {
        "id": f"rs_fixture_{index}",
        "type": "reasoning",
        "summary": [],
        "encrypted_content": f"synthetic-opaque-fixture-{index}",
        "status": "completed",
    }


def function_call(index: int, *, arguments: str | None = None) -> dict:
    return {
        "id": f"fc_fixture_{index}",
        "call_id": f"call_fixture_{index}",
        "type": "function_call",
        "name": "fixture_lookup",
        "arguments": arguments or f'{{"record":"fixture-{index}"}}',
    }


def assistant_message(index: int, *, text: str | None = None) -> dict:
    return {
        "id": f"msg_fixture_{index}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": text or f"Synthetic message {index}.",
                "annotations": [],
            }
        ],
    }


def replay_items(output: list[dict]) -> list[dict]:
    message = Message.from_llm_responses_output(output)
    return _assistant_to_responses_items(message)


def events_for_tool_message(message: Message) -> list[ActionEvent]:
    assert message.tool_calls
    return [
        ActionEvent(
            thought=message.content if index == 0 else [],
            reasoning_content=message.reasoning_content if index == 0 else None,
            thinking_blocks=list(message.thinking_blocks) if index == 0 else [],
            responses_reasoning_item=(
                message.responses_reasoning_item if index == 0 else None
            ),
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            tool_call=tool_call,
            llm_response_id="resp_fixture_batch",
        )
        for index, tool_call in enumerate(message.tool_calls)
    ]


class ResponsesHistoryTests(unittest.TestCase):
    def test_all_reasoning_items_survive_replay(self):
        output = [reasoning(1), reasoning(2), reasoning(3), function_call(1)]
        self.assertEqual(replay_items(output), output)

    def test_reasoning_only_response_preserves_every_item(self):
        output = [reasoning(1), reasoning(2)]
        self.assertEqual(replay_items(output), output)

    def test_interleaved_items_keep_original_order_and_identity(self):
        output = [
            reasoning(1),
            assistant_message(1, text="First synthetic message."),
            function_call(1, arguments='{ "z": 2, "a": "line\\nvalue" }'),
            reasoning(2),
            assistant_message(2, text="Second synthetic message."),
            function_call(2),
        ]
        self.assertEqual(replay_items(output), output)

    def test_multi_tool_event_reload_preserves_reasoning_batch(self):
        output = [reasoning(1), function_call(1), function_call(2)]
        events = events_for_tool_message(Message.from_llm_responses_output(output))
        reloaded = [
            ActionEvent.model_validate_json(event.model_dump_json()) for event in events
        ]
        combined = LLMConvertibleEvent.events_to_messages(reloaded)
        self.assertEqual(len(combined), 1)
        self.assertEqual(_assistant_to_responses_items(combined[0]), output)

    def test_message_without_capture_falls_back_to_reconstruction(self):
        # No reasoning item -> nothing to carry the capture; the SDK's own
        # serialization applies unchanged.
        output = [assistant_message(1, text="Plain answer."), function_call(1)]
        message = Message.from_llm_responses_output(output)
        items = _assistant_to_responses_items(message)
        self.assertEqual(
            [item["type"] for item in items], ["message", "function_call"]
        )


if __name__ == "__main__":
    unittest.main()
