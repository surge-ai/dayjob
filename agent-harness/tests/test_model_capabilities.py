"""Contracts for install_model_capabilities (reasoning replay + registry gaps)."""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest  # noqa: E402
from openhands.sdk import LLM  # noqa: E402
from openhands.sdk.llm.utils import model_features as _model_features  # noqa: E402

from agent_harness.openhands_patches import (  # noqa: E402
    REASONING_REPLAY_MODELS,
    install_model_capabilities,
)

# The SDK matches replay patterns as case-insensitive substrings of the full
# model id, so any pattern that is itself a fragment of a provider or proxy
# name ("gemini-", "gpt/", "lite") would enable replay for that provider's
# (or every proxied) model.
_PROVIDER_NAMES = (
    "anthropic",
    "azure",
    "bedrock",
    "claude",
    "gemini",
    "gpt",
    "litellm_proxy",
    "openai",
    "openrouter",
    "vertex_ai",
)


@pytest.mark.parametrize("pattern", REASONING_REPLAY_MODELS)
def test_no_dangerously_broad_replay_patterns(pattern):
    normalized = pattern.casefold().rstrip("/-_.: ")
    assert normalized, f"empty replay pattern {pattern!r}"
    assert not any(normalized in name for name in _PROVIDER_NAMES), (
        f"replay pattern {pattern!r} is a fragment of a provider name and "
        "would enable replay for that whole provider"
    )


def test_install_is_idempotent(monkeypatch):
    allowlist = ["existing/model"]
    monkeypatch.setattr(
        _model_features, "SEND_REASONING_CONTENT_MODELS", allowlist
    )

    install_model_capabilities()
    install_model_capabilities()

    assert allowlist == ["existing/model", *REASONING_REPLAY_MODELS]


@pytest.mark.parametrize(
    "model",
    [
        "openrouter/qwen/qwen3.8-max",
        "litellm_proxy/tencent/hy3-thinking",
        "z-ai/glm-5",
        "meta/muse-glimmer-30b-v2",
        "thinkingmachines/inkling-preview",
    ],
)
def test_configured_models_send_reasoning_content(model):
    install_model_capabilities()

    assert _model_features.get_features(model).send_reasoning_content


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-5",
        "openrouter/anthropic/claude-sonnet-4",
        "qwen/qwen3.8-coder",
        "tencent/hy4",
        "meta/muse-glimmer-29b",
    ],
)
def test_unconfigured_models_do_not_send_reasoning_content(model):
    install_model_capabilities()

    assert not _model_features.get_features(model).send_reasoning_content


@pytest.mark.parametrize(
    "model",
    ["gemini/gemini-3.1-pro-preview", "gemini/gemini-3.7-flash", "gemini-3.7-flash"],
)
def test_gemini_models_keep_native_reasoning_effort(model):
    """Registering the missing id with litellm makes the SDK detect
    supports_reasoning_effort, so the param reaches the request."""
    install_model_capabilities()

    llm = LLM(model=model, reasoning_effort="max")
    assert llm._model_features().supports_reasoning_effort
    call_kwargs = llm._finalize_completion_params([], None, False, {})[3]
    assert call_kwargs["reasoning_effort"] == "max"
