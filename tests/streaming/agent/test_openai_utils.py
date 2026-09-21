import pytest
from pydantic.v1 import BaseModel
from vocode.streaming.agent.openai_utils import (
    format_openai_chat_messages_from_transcript,
    get_openai_chat_messages_from_transcript,
)
from vocode.streaming.agent.token_utils import num_tokens_from_messages
from vocode.streaming.models.actions import (
    ACTION_FINISHED_FORMAT_STRING,
    ActionConfig,
    ActionInput,
    ActionOutput,
    PhraseBasedActionTrigger,
    PhraseBasedActionTriggerConfig,
)
from vocode.streaming.models.agent import LLM_AGENT_DEFAULT_MAX_TOKENS
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import (
    ActionFinish,
    ActionStart,
    Message,
    Transcript,
)


class WeatherActionConfig(ActionConfig, type="weather"):
    pass


class WeatherParameters(BaseModel):
    location: str


def create_fake_vocode_phrase_trigger():
    return PhraseBasedActionTrigger(config=PhraseBasedActionTriggerConfig(phrase_triggers=[]))


def test_format_openai_chat_messages_from_transcript():
    """Verification: Transcript formatting preserves messages and pairs only actual model-issued tool calls."""
    test_action_input_nophrase = ActionInput(
        action_config=WeatherActionConfig(),
        conversation_id="asdf",
        params={},
    )
    test_action_input_phrase = ActionInput(
        action_config=WeatherActionConfig(action_trigger=create_fake_vocode_phrase_trigger()),
        conversation_id="asdf",
        params={},
    )

    test_cases = [
        (
            (
                Transcript(
                    event_logs=[
                        Message(sender=Sender.BOT, text="Hello!", is_final=True),
                        Message(
                            sender=Sender.BOT,
                            text="How are you doing today?",
                            is_final=True,
                        ),
                        Message(sender=Sender.HUMAN, text="I'm doing well, thanks!"),
                    ]
                ),
                "gpt-3.5-turbo-0613",
                None,
                "prompt preamble",
            ),
            [
                {"role": "system", "content": "prompt preamble"},
                {"role": "assistant", "content": "Hello! How are you doing today?"},
                {"role": "user", "content": "I'm doing well, thanks!"},
            ],
        ),
        (
            (
                Transcript(
                    event_logs=[
                        Message(sender=Sender.BOT, text="Hello!", is_final=True),
                        Message(sender=Sender.BOT, text="How are", is_final=False),
                    ]
                ),
                "gpt-3.5-turbo-0613",
                None,
                "prompt preamble",
            ),
            [
                {"role": "system", "content": "prompt preamble"},
                {"role": "assistant", "content": "Hello! How are-"},
            ],
        ),
        (
            (
                Transcript(
                    event_logs=[
                        Message(sender=Sender.BOT, text="Hello!", is_final=True),
                        Message(
                            sender=Sender.HUMAN, text="Hello, what's the weather like?"
                        ),
                        ActionStart(
                            action_type="weather",
                            action_input=test_action_input_nophrase,
                            tool_call_id="tool_weather",
                        ),
                        ActionFinish(
                            action_type="weather",
                            action_input=test_action_input_nophrase,
                            action_output=ActionOutput(
                                action_type="weather", response={}
                            ),
                            tool_call_id="tool_weather",
                        ),
                    ]
                ),
                "gpt-3.5-turbo-0613",
                None,
                "some prompt",
            ),
            [
                {"role": "system", "content": "some prompt"},
                {"role": "assistant", "content": "Hello!"},
                {
                    "role": "user",
                    "content": "Hello, what's the weather like?",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tool_weather",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tool_weather",
                    "content": ACTION_FINISHED_FORMAT_STRING.format(
                        action_name="weather", action_output="{}"
                    ),
                },
            ],
        ),
        (
            (
                Transcript(
                    event_logs=[
                        Message(sender=Sender.BOT, text="Hello!", is_final=True),
                        Message(
                            sender=Sender.HUMAN, text="Hello, what's the weather like?"
                        ),
                        ActionStart(
                            action_type="weather",
                            action_input=test_action_input_phrase,
                        ),
                        ActionFinish(
                            action_type="weather",
                            action_input=test_action_input_phrase,
                            action_output=ActionOutput(
                                action_type="weather", response={}
                            ),
                        ),
                    ]
                ),
                "gpt-3.5-turbo-0613",
                None,
                "some prompt",
            ),
            [
                {"role": "system", "content": "some prompt"},
                {"role": "assistant", "content": "Hello!"},
                {
                    "role": "user",
                    "content": "Hello, what's the weather like?",
                },
            ],
        ),
    ]

    for params, expected_output in test_cases:
        assert format_openai_chat_messages_from_transcript(*params) == expected_output


@pytest.mark.parametrize("keep_recent", [False, True])
def test_context_trimming_removes_old_tool_pair_and_preserves_recent_history(
    monkeypatch, keep_recent
):
    """Critical: Oversized tool arguments evict their whole exchange without orphaning responses or losing recent history."""
    transcript = Transcript()
    for call_id in ["old", "recent"] if keep_recent else ["old"]:
        action_input = ActionInput(
            action_config=WeatherActionConfig(),
            conversation_id="weather_conversation",
            params=WeatherParameters(
                location="Paris " * (1000 if call_id == "old" else 1)
            ),
        )
        transcript.event_logs.extend(
            [
                ActionStart(
                    action_type="weather",
                    action_input=action_input,
                    tool_call_id=call_id,
                ),
                ActionFinish(
                    action_type="weather",
                    action_input=action_input,
                    action_output=ActionOutput(action_type="weather", response={}),
                    tool_call_id=call_id,
                ),
            ]
        )
    if keep_recent:
        transcript.event_logs.append(
            Message(sender=Sender.HUMAN, text="What should I wear?")
        )
    model, preamble = "gpt-4o-mini", "Help with the weather."
    original = get_openai_chat_messages_from_transcript(transcript.event_logs, preamble)
    assert original[1]["tool_calls"][0]["function"]["arguments"].count("Paris") == 1000
    expected = [original[0], *original[3:]]
    budget = num_tokens_from_messages(expected, model) + 64
    monkeypatch.setattr(
        "vocode.streaming.agent.openai_utils.get_chat_gpt_max_tokens",
        lambda _: budget + LLM_AGENT_DEFAULT_MAX_TOKENS + 50,
    )
    actual = format_openai_chat_messages_from_transcript(
        transcript, model, None, preamble
    )
    assert actual == expected
    assert num_tokens_from_messages(actual, model) <= budget
    assert len(transcript.event_logs) == (5 if keep_recent else 2)


def test_context_trimming_removes_all_responses_for_parallel_tool_calls(monkeypatch):
    """Edge Case: Evicting one multi-tool declaration removes every matching response together."""
    messages = [
        {"role": "system", "content": "Help with the weather."},
        {
            "role": "assistant",
            "content": "Checking both locations. " * 1000,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                }
                for call_id in ("paris", "london")
            ],
        },
        {"role": "tool", "tool_call_id": "london", "content": "Rainy"},
        {"role": "tool", "tool_call_id": "paris", "content": "Sunny"},
    ]
    expected = messages[:1]
    monkeypatch.setattr(
        "vocode.streaming.agent.openai_utils.get_openai_chat_messages_from_transcript",
        lambda **_: messages.copy(),
    )
    monkeypatch.setattr(
        "vocode.streaming.agent.openai_utils.get_chat_gpt_max_tokens",
        lambda _: (
            num_tokens_from_messages(expected) + 64 + LLM_AGENT_DEFAULT_MAX_TOKENS + 50
        ),
    )
    assert (
        format_openai_chat_messages_from_transcript(
            Transcript(), "gpt-4o-mini", None, ""
        )
        == expected
    )


def test_format_openai_chat_messages_from_transcript_context_limit():
    """Edge Case: Stored legacy transcripts trim old messages while retaining their system prompt."""
    test_cases = [
        (
            (
                Transcript(
                    event_logs=[
                        Message(sender=Sender.BOT, text="Hello!", is_final=True),
                        Message(
                            sender=Sender.BOT,
                            text="How are you doing today? I'm doing amazing thank you so much for asking!",
                        ),
                        Message(sender=Sender.HUMAN, text="I'm doing well, thanks!"),
                    ]
                ),
                "gpt-3.5-turbo-0613",
                None,
                "aaaa " * 1862,
            ),
            [
                {"role": "system", "content": "aaaa " * 1862},
                {"role": "user", "content": "I'm doing well, thanks!"},
            ],
        ),
        (
            (
                Transcript(
                    event_logs=[
                        Message(sender=Sender.BOT, text="Hello!"),
                        Message(
                            sender=Sender.BOT,
                            text="How are you doing today? I'm doing amazing thank you so much for asking!",
                            is_final=True,
                        ),
                        Message(sender=Sender.HUMAN, text="I'm doing well, thanks!"),
                        Message(sender=Sender.BOT, text="aaaa " * 1862),
                        Message(
                            sender=Sender.HUMAN, text="What? What did you just say???"
                        ),
                        Message(
                            sender=Sender.BOT,
                            text="My apologies, there was an error. Please ignore my previous message",
                            is_final=True,
                        ),
                        Message(
                            sender=Sender.HUMAN,
                            text="Don't worry I ignored all 1862 * 5 characters of it.",
                        ),
                    ]
                ),
                "gpt-3.5-turbo-0613",
                None,
                "prompt preamble",
            ),
            [
                {"role": "system", "content": "prompt preamble"},
                {
                    "content": "What? What did you just say???",
                    "role": "user",
                },
                {
                    "role": "assistant",
                    "content": "My apologies, there was an error. Please ignore my previous message",
                },
                {
                    "role": "user",
                    "content": "Don't worry I ignored all 1862 * 5 characters of it.",
                },
            ],
        ),
    ]

    for params, expected_output in test_cases:
        assert format_openai_chat_messages_from_transcript(*params) == expected_output
