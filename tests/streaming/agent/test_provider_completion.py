import time
from types import SimpleNamespace

import pytest
from google.genai import types
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from vocode.streaming.agent.chat_gpt_agent import instantiate_openai_client
from vocode.streaming.agent.gemini_agent import GeminiAgent
from vocode.streaming.agent.openai_utils import openai_get_tokens
from vocode.streaming.utils.provider_lifecycle import (
    ProviderScope,
    current_provider_scope,
    provider_request,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "gemini"])
@pytest.mark.parametrize("finished", [False, True])
async def test_explicit_provider_final_reason_is_required(provider, finished):
    """Critical: Stream EOF without a provider terminal reason cannot confirm inference completion."""
    owner = ProviderScope(deadline=time.monotonic() + 10)

    async def chunks():
        if provider == "openai":
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "test",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "text"},
                            "finish_reason": "stop" if finished else None,
                        }
                    ],
                }
            )
        else:
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(parts=[types.Part(text="text")]),
                        finish_reason=types.FinishReason.STOP if finished else None,
                    )
                ]
            )

    with provider_request(owner) as request:
        generator = (
            openai_get_tokens(chunks(), provider_request=request)
            if provider == "openai"
            else GeminiAgent._token_generator(None, chunks(), provider_request=request)
        )
        [value async for value in generator]
    assert owner.pending == 0 and owner.uncertain is not finished


@pytest.mark.parametrize("managed", [False, True])
def test_managed_openai_cannot_hide_retried_requests(mocker, managed):
    """Critical: Managed inference disables hidden SDK retries while ordinary calls retain their default."""
    constructor = mocker.patch("vocode.streaming.agent.chat_gpt_agent.AsyncOpenAI")
    owner = ProviderScope(deadline=time.monotonic() + 10) if managed else None
    token = current_provider_scope.set(owner)
    try:
        instantiate_openai_client(
            SimpleNamespace(
                azure_params=None, openai_api_key="test", base_url_override=None
            )
        )
    finally:
        current_provider_scope.reset(token)
    retries = constructor.call_args.kwargs["max_retries"]
    assert retries == 0 if managed else retries > 0
