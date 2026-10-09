"""LiteLLM-internal kwargs must not be serialized into OpenAI-shaped request bodies.

transform_request() spreads optional_params into the body, and strict providers
(Bedrock Mantle) 400 the whole request on an unknown field such as model_alias_map.
"""

import asyncio

import pytest

from litellm.litellm_core_utils.chat_completion_agentic_loop import _FOLLOWUP_INTERNAL_PARAMS
from litellm.llms.bedrock_mantle.chat.transformation import BedrockMantleChatConfig
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

INTERNAL = {"model_alias_map": {"alias": {"model": "target"}}, "custom_prompt_dict": {}, "stream_response": {}}
REAL = {
    "temperature": 0.4,
    "max_tokens": 99,
    "stream": True,
    "stream_options": {"include_usage": True},
    "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}],
}
MESSAGES = [{"role": "user", "content": "x"}]
KWARGS = {"model": "global.openai.gpt-6-astra", "litellm_params": {"custom_llm_provider": "bedrock_mantle"}, "headers": {}}


def _assert_clean(body: dict) -> None:
    assert not set(INTERNAL) & set(body)
    for name, value in REAL.items():
        assert body.get(name) == value, name


@pytest.mark.parametrize("config", [OpenAIGPTConfig(), BedrockMantleChatConfig()], ids=lambda c: type(c).__name__)
def test_transform_request_drops_internal_params(config):
    _assert_clean(config.transform_request(messages=list(MESSAGES), optional_params={**INTERNAL, **REAL}, **KWARGS))


def test_async_transform_request_drops_internal_params(monkeypatch):
    config = OpenAIGPTConfig()
    monkeypatch.setattr(type(config), "_is_base_class", True, raising=False)
    body = asyncio.run(config.async_transform_request(messages=list(MESSAGES), optional_params={**INTERNAL, **REAL}, **KWARGS))
    _assert_clean(body)


def test_only_upstream_declared_internals_are_dropped():
    assert set(INTERNAL) <= set(_FOLLOWUP_INTERNAL_PARAMS)
