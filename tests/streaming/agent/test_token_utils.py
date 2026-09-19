import pytest

from vocode.streaming.agent.token_utils import (
    CHAT_GPT_MAX_TOKENS,
    get_chat_gpt_max_tokens,
    num_tokens_from_messages,
)


@pytest.mark.parametrize("model", [*CHAT_GPT_MAX_TOKENS, "gpt-3.5-turbo-0613"])
def test_declared_and_legacy_models_have_usable_token_accounting(model):
    """Verification: Declared models and stored legacy transcripts can be counted without invoking providers."""
    count = num_tokens_from_messages([{"role": "user", "content": "Hello"}], model)
    assert 0 < count < get_chat_gpt_max_tokens(model)
