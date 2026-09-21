"""ATIF trajectory building (harbor's Agent Trajectory Interchange Format),
adapted from harbor's reference OpenHands SDK agent
(``harbor/agents/installed/openhands_sdk_runner.py``).

Differences from the reference: SDK events convert to ATIF steps in a single
pass; agent ``thought`` text on ``ActionEvent`` becomes the step's ``message``
(the reference drops inter-tool commentary); and the document carries an
``extra`` field — ATIF's designated slot for custom metadata — holding the
harness's verdict (``TrajectoryExtras``).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from openhands.sdk.event import (
    ActionEvent,
    MessageEvent,
    ObservationEvent,
)

ATIF_SCHEMA_VERSION = "ATIF-v1.5"


def _text_of(content: Any) -> str:
    """Join the text parts of an SDK message/observation content value."""
    if isinstance(content, list):
        return "\n".join(
            getattr(c, "text", str(c)) for c in content if getattr(c, "text", None)
        )
    if content:
        return str(content)
    return ""


def _action_arguments(event: ActionEvent) -> dict[str, Any]:
    """Tool-call arguments, preferring the raw tool call and falling back to
    the parsed action's fields. The SDK's ``MessageToolCall`` carries
    ``arguments`` directly; OpenAI-format calls nest them in ``function``."""
    if event.tool_call is not None:
        raw_args = getattr(
            getattr(event.tool_call, "function", event.tool_call), "arguments", None
        )
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError:
                return {"raw": raw_args}
    if event.action is not None:
        try:
            action_dict = (
                event.action.model_dump()
                if hasattr(event.action, "model_dump")
                else vars(event.action)
            )
            return {
                k: v for k, v in action_dict.items() if k != "kind" and v is not None
            }
        except Exception:
            pass
    return {}


def events_to_steps(events: Any, model_name: str) -> list[dict[str, Any]]:
    """Convert SDK conversation events to ATIF steps (without step IDs).
    Other event types (condensation, system prompt, agent errors, ...) are
    skipped, as in the reference implementation."""
    steps: list[dict[str, Any]] = []

    for event in events:
        if isinstance(event, MessageEvent) and event.source in ("user", "agent"):
            steps.append(
                {
                    "timestamp": event.timestamp,
                    "source": event.source,
                    "message": _text_of(getattr(event.llm_message, "content", None)),
                }
            )
        elif isinstance(event, ActionEvent):
            steps.append(
                {
                    "timestamp": event.timestamp,
                    "source": "agent",
                    "message": _text_of(getattr(event, "thought", None)),
                    "model_name": model_name,
                    "tool_calls": [
                        {
                            "tool_call_id": event.tool_call_id,
                            "function_name": event.tool_name,
                            "arguments": _action_arguments(event),
                        }
                    ],
                }
            )
        elif isinstance(event, ObservationEvent):
            # Attach the observation to the step holding its tool call. A
            # multi-tool-call response arrives as consecutive ActionEvents,
            # so the answering step is not necessarily the latest one.
            step = next(
                (
                    s
                    for s in reversed(steps)
                    if any(
                        c["tool_call_id"] == event.tool_call_id
                        for c in s.get("tool_calls", ())
                    )
                ),
                None,
            )
            if step is not None:
                content = _text_of(getattr(event.observation, "content", None)) or str(
                    event.observation
                )
                step.setdefault("observation", {"results": []})["results"].append(
                    {"source_call_id": event.tool_call_id, "content": content}
                )

    for step in steps:
        if step["source"] == "agent":
            step.setdefault("model_name", model_name)
    return steps


def build_trajectory(
    steps: list[dict[str, Any]],
    llm_metrics: dict[str, Any],
    agent_name: str,
    agent_version: str,
    model_name: str | None = None,
    system_prompt: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ATIF document from converted steps."""
    if system_prompt:
        steps = [
            {
                "timestamp": steps[0]["timestamp"] if steps else None,
                "source": "system",
                "message": system_prompt,
            },
            *steps,
        ]
    for i, step in enumerate(steps):
        step["step_id"] = i + 1

    trajectory: dict[str, Any] = {
        "schema_version": ATIF_SCHEMA_VERSION,
        # Required through ATIF v1.6 (optional only from v1.7). As in the
        # reference runner, prefer the host-provided run id.
        "session_id": os.environ.get("SESSION_ID") or str(uuid.uuid4()),
        "agent": {
            "name": agent_name,
            "version": agent_version,
            "model_name": model_name,
            "tool_definitions": tool_definitions if tool_definitions else None,
        },
        "steps": steps,
        # All final_metrics fields are optional in ATIF; omit unknown values
        # rather than fabricating zeros (None means the SDK metrics API gave
        # us nothing, which downstream reporting treats differently from 0).
        "final_metrics": {
            key: value
            for key, value in {
                "total_prompt_tokens": llm_metrics.get("prompt_tokens"),
                "total_completion_tokens": llm_metrics.get("completion_tokens"),
                "total_cached_tokens": llm_metrics.get("cached_tokens"),
                "total_cost_usd": llm_metrics.get("cost_usd"),
            }.items()
            if value is not None
        },
    }
    if extra is not None:
        trajectory["extra"] = extra
    return trajectory
