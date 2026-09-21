"""Condenser transport contracts against the pinned OpenHands SDK (offline).

The SDK's LLMSummarizingCondenser hard-codes the Chat Completions transport;
``configure_condenser_transport`` reroutes it to the Responses API for
Responses-API models. All LLM transports are stubbed, so no network I/O.
"""

import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from openhands.sdk import LLM, LLMSummarizingCondenser  # noqa: E402
from openhands.sdk.context.view import View  # noqa: E402
from openhands.sdk.event import MessageEvent  # noqa: E402
from openhands.sdk.llm import Message, TextContent  # noqa: E402

from agent_harness.openhands_patches import (  # noqa: E402
    configure_condenser_transport,
)


def _event_view() -> View:
    # One past the condenser's default max_size (240) forces condensation.
    events = [
        MessageEvent(
            id=f"event-fixture-{index}",
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text=f"Synthetic event {index}")],
            ),
        )
        for index in range(241)
    ]
    return View.from_events(events)


def _instrument_llm(model: str, **llm_kwargs):
    """Build an LLM with all four transports stubbed, then configure it."""
    llm = LLM(
        model=model,
        api_key="condenser-contract-fixture-not-a-real-key",
        num_retries=0,
        timeout=1,
        **llm_kwargs,
    )
    calls: list[str] = []

    def _response():
        return SimpleNamespace(
            id="resp_condenser_fixture",
            message=Message(
                role="assistant",
                content=[TextContent(text="Synthetic condensation summary.")],
            ),
        )

    def _sync_stub(name):
        def stub(*args, **kwargs):
            calls.append(name)
            return _response()

        return stub

    def _async_stub(name):
        async def stub(*args, **kwargs):
            calls.append(name)
            return _response()

        return stub

    object.__setattr__(llm, "completion", _sync_stub("completion"))
    object.__setattr__(llm, "responses", _sync_stub("responses"))
    object.__setattr__(llm, "acompletion", _async_stub("acompletion"))
    object.__setattr__(llm, "aresponses", _async_stub("aresponses"))
    configure_condenser_transport(llm)
    return llm, calls


def test_chat_condenser_stays_on_chat_completions():
    llm, calls = _instrument_llm("openai/gpt-4.1", api_mode="chat")
    assert not llm.uses_responses_api()
    LLMSummarizingCondenser(llm=llm).condense(_event_view())
    assert calls == ["completion"]


def test_detected_responses_condenser_uses_responses():
    # gpt-5* models are auto-detected as Responses-API models (api_mode "auto").
    llm, calls = _instrument_llm("openai/gpt-5.6-sol")
    assert llm.uses_responses_api()
    LLMSummarizingCondenser(llm=llm).condense(_event_view())
    assert calls == ["responses"]


def test_forced_responses_condenser_uses_responses():
    llm, calls = _instrument_llm("openai/muse-spark-1.2", api_mode="responses")
    LLMSummarizingCondenser(llm=llm).condense(_event_view())
    assert calls == ["responses"]


def test_async_responses_condenser_uses_async_responses():
    llm, calls = _instrument_llm("openai/grok-4.6", api_mode="responses")
    asyncio.run(LLMSummarizingCondenser(llm=llm).acondense(_event_view()))
    assert calls == ["aresponses"]


def test_responses_condenser_omits_tool_choice():
    llm, _calls = _instrument_llm("openai/grok-4.6", api_mode="responses")
    messages = [Message(role="user", content=[TextContent(text="Summarize.")])]

    sync_prepared = llm._prepare_responses_params(messages, None, None, None, False, {})
    async_prepared = asyncio.run(
        llm._aprepare_responses_params(messages, None, None, None, False, {})
    )
    for _instructions, _input, tools, call_kwargs, _telemetry in (
        sync_prepared,
        async_prepared,
    ):
        assert tools is None
        assert "tool_choice" not in call_kwargs
