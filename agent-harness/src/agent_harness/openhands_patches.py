"""Compatibility patches for the pinned OpenHands SDK.

These patches depend on SDK internals. Run ``test_responses_history`` and
``test_condenser_transport`` when updating the SDK version in ``pyproject.toml``.
"""

from __future__ import annotations

import copy
from typing import Any

import litellm
import openhands.sdk.llm.message as _message
import openhands.sdk.llm.utils.responses_serialization as ser_mod
import openhands.sdk.utils.command as _command
import openhands.sdk.utils.truncate as _truncate
from openhands.sdk.event.base import Event
from openhands.sdk.llm.message import Message, ReasoningItemModel
from openhands.sdk.llm.utils import model_features as _model_features
from pydantic.fields import FieldInfo

from agent_harness import LLM_CREDENTIAL_ENV_VARS

_responses_replay_installed = False

# Additional reasoning-replay models, matched as substrings of model IDs.
# Remove entries as the pinned SDK adds support.
REASONING_REPLAY_MODELS = (
    "qwen/qwen3.8-max",
    "tencent/hy3",
    "z-ai/glm",
    "meta/muse-glimmer-30b",
    "thinkingmachines/inkling",
    "meta/muse-spark-1.2",
    "meta/muse-spark-1.3",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "tencent/hy4-preview",
    "kimi-k2.7",
    # Replay plaintext reasoning on the OpenAI-compatible route. The pinned
    # SDK drops encrypted thought signatures, so those cannot be replayed.
    "google/gemini-3.8-flash",
)

# Model metadata missing from the pinned LiteLLM registry.
# Remove these overrides when the pinned registry includes them.
_GEMINI_37_FLASH_INFO = {
    "litellm_provider": "gemini",
    "mode": "chat",
    "supports_reasoning": True,
    "supports_function_calling": True,
    "supports_vision": True,
    "supports_prompt_caching": True,
}
MISSING_LITELLM_MODELS = {
    "gemini/gemini-3.7-flash": _GEMINI_37_FLASH_INFO,
    "gemini-3.7-flash": _GEMINI_37_FLASH_INFO,
}


def install_model_capabilities() -> None:
    """Fix model-capability gaps in the pinned SDK/litellm. Idempotent."""
    allowlist = _model_features.SEND_REASONING_CONTENT_MODELS
    for pattern in REASONING_REPLAY_MODELS:
        if pattern not in allowlist:
            allowlist.append(pattern)
    litellm.register_model(MISSING_LITELLM_MODELS)


def hide_credentials_from_tools() -> None:
    """Add the forwarded LLM credentials to the SDK's ``sanitized_env()``
    denylist, which every SDK-spawned subprocess builds its environment from.

    WARNING: this does not work if tmux is installed in the image. The SDK
    then picks the tmux terminal backend, which bypasses ``sanitized_env()``
    (libtmux ignores ``Server(environment=...)``, so every pane inherits the
    runner's full process environment). Our images do not install tmux.
    """
    _command._SENSITIVE_ENV_VARS = _command._SENSITIVE_ENV_VARS | frozenset(
        LLM_CREDENTIAL_ENV_VARS
    )


def install_responses_replay() -> None:
    """Patch the SDK so Responses API history survives round-trips.

    The SDK retains only the last reasoning item when converting a response
    to a Message. Store the full output on that item and replay it in order.
    Safe to call more than once; install before starting the agent loop.
    """
    global _responses_replay_installed
    if _responses_replay_installed:
        return

    # Add a field that survives conversation persistence.
    if "responses_output_items" not in ReasoningItemModel.__pydantic_fields__:
        ReasoningItemModel.__pydantic_fields__["responses_output_items"] = FieldInfo(
            annotation=list[dict[str, Any]] | None, default=None, repr=False
        )
        ReasoningItemModel.model_rebuild(force=True)

        # Nested schemas were compiled against the old shape; recompile every
        # model that can carry a ReasoningItemModel through persistence.
        Message.model_rebuild(force=True)
        _rebuild_subclass_tree(Event)

    # Capture the full output before the SDK discards reasoning items.
    original_from_output = Message.from_llm_responses_output.__func__

    def from_output_with_capture(cls: type, output: Any) -> Any:
        items = list(output or [])
        message = original_from_output(cls, items)
        carrier = message.responses_reasoning_item
        if carrier is not None:
            carrier.responses_output_items = [
                copy.deepcopy(item)
                if isinstance(item, dict)
                else item.model_dump(mode="json", exclude_none=True)
                for item in items
            ]
        return message

    Message.from_llm_responses_output = classmethod(from_output_with_capture)

    # Replay the captured output when serializing assistant messages.
    original_assistant = ser_mod._assistant_to_responses_items

    def assistant_with_replay(message: Any) -> list[dict[str, Any]]:
        """Replay the original items verbatim on serialization."""
        carrier = getattr(message, "responses_reasoning_item", None)
        items = getattr(carrier, "responses_output_items", None)
        if items is not None:
            return copy.deepcopy(items)
        return original_assistant(message)

    ser_mod._assistant_to_responses_items = assistant_with_replay

    _responses_replay_installed = True


def _rebuild_subclass_tree(cls: type) -> None:
    """Rebuild the Pydantic model tree for all subclasses of the given class."""
    cls.model_rebuild(force=True)  # type: ignore[attr-defined]
    for subclass in cls.__subclasses__():
        _rebuild_subclass_tree(subclass)


def persist_large_tool_outputs(save_dir: str) -> None:
    """Save complete tool output when the SDK clips it for the LLM."""

    def truncate(_: _message.Message, text: str) -> str:
        return _truncate.maybe_truncate(
            text,
            truncate_after=_message.DEFAULT_TEXT_CONTENT_LIMIT,
            save_dir=save_dir,
            tool_prefix="tool",
        )

    _message.Message._maybe_truncate_tool_text = truncate


def configure_condenser_transport(llm: Any) -> None:
    """Route condenser calls through the Responses API when the model uses it.

    The SDK's ``LLMSummarizingCondenser`` hard-codes the Chat Completions
    transport (``completion``/``acompletion``) even when its model is a
    Responses-API model, so a trial that triggers compaction sends
    Responses-specific request configuration to the wrong endpoint and
    fails. Rebind the condenser LLM's transports to
    ``responses``/``aresponses``; condenser calls are tool-free, so drop
    ``tool_choice`` from the prepared request.
    """
    if not llm.uses_responses_api():
        return

    finalize = llm._finalize_responses_params

    def finalize_condenser_params(*args: Any, **kwargs: Any) -> Any:
        # (instructions, input_items, resp_tools, call_kwargs, telemetry_ctx)
        prepared = finalize(*args, **kwargs)
        prepared[3].pop("tool_choice", None)
        telemetry_kwargs = prepared[4].get("kwargs")
        if isinstance(telemetry_kwargs, dict):
            telemetry_kwargs.pop("tool_choice", None)
        return prepared

    # LLM is a frozen pydantic model; bypass its setattr guard.
    object.__setattr__(llm, "_finalize_responses_params", finalize_condenser_params)
    object.__setattr__(llm, "completion", llm.responses)
    object.__setattr__(llm, "acompletion", llm.aresponses)
