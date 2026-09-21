import json
from copy import deepcopy

import pytest
from vocode.streaming.agent.token_utils import (
    CHAT_GPT_MAX_TOKENS,
    get_chat_gpt_max_tokens,
    get_tokenizer_info,
    num_tokens_from_messages,
)


@pytest.mark.parametrize("model", [*CHAT_GPT_MAX_TOKENS, "gpt-3.5-turbo-0613"])
def test_declared_and_legacy_models_have_usable_token_accounting(model):
    """Verification: Declared models and stored legacy transcripts can be counted without invoking providers."""
    count = num_tokens_from_messages([{"role": "user", "content": "Hello"}], model)
    assert 0 < count < get_chat_gpt_max_tokens(model)


@pytest.mark.parametrize("call_count", [1, 2])
def test_tool_arguments_contribute_to_context_size(call_count):
    """Critical: Every tool call's serialized arguments contribute to the context estimate."""
    model = "gpt-4o-mini"
    arguments = json.dumps({"locations": ["Paris"] * 1000})
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"weather_{index}",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                }
                for index in range(call_count)
            ],
        }
    ]
    expanded = deepcopy(messages)
    for call in expanded[0]["tool_calls"]:
        call["function"]["arguments"] = arguments
    encoding = get_tokenizer_info(model).encoding
    expected_increase = call_count * (
        len(encoding.encode(arguments)) - len(encoding.encode("{}"))
    )
    assert (
        num_tokens_from_messages(expanded, model)
        - num_tokens_from_messages(messages, model)
        == expected_increase
    )
