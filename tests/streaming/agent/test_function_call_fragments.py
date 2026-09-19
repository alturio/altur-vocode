import pytest
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from vocode.streaming.agent.openai_utils import openai_get_tokens
from vocode.streaming.agent.streaming_utils import (
    collate_response_async,
    stream_response_async,
)
from vocode.streaming.models.actions import FunctionCall


@pytest.mark.asyncio
@pytest.mark.parametrize("streamer", [collate_response_async, stream_response_async])
@pytest.mark.parametrize("modern", [False, True])
async def test_fragmented_function_names_are_preserved(streamer, modern):
    """Critical: Legacy and current provider deltas produce the complete tool name on both output paths."""

    async def chunks():
        for name, arguments in (("wave", "{"), ("_hello", '"name":"user"}')):
            function = {"name": name, "arguments": arguments}
            delta = (
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_test",
                            "type": "function",
                            "function": function,
                        }
                    ]
                }
                if modern
                else {"function_call": function}
            )
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "test",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
            )

    result = [
        item
        async for item in streamer(
            conversation_id="test", gen=openai_get_tokens(chunks()), get_functions=True
        )
    ]
    assert result == [
        FunctionCall(
            name="wave_hello",
            arguments='{"name":"user"}',
            tool_call_id="call_test" if modern else None,
        )
    ]
