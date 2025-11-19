from copy import deepcopy
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from loguru import logger
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from vocode.streaming.agent.token_utils import (
    get_chat_gpt_max_tokens,
    num_tokens_from_functions,
    num_tokens_from_messages,
)
from vocode.streaming.models.actions import FunctionFragment, PhraseBasedActionTrigger
from vocode.streaming.models.agent import LLM_AGENT_DEFAULT_MAX_TOKENS
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import (
    ActionFinish,
    ActionStart,
    ConferenceEvent,
    EventLog,
    Message,
    Transcript,
)


def vector_db_result_to_openai_chat_message(vector_db_result):
    return {"role": "user", "content": vector_db_result}


def is_phrase_based_action_event_log(event_log: EventLog) -> bool:
    return (
        (isinstance(event_log, ActionStart) or isinstance(event_log, ActionFinish))
        and event_log.action_input is not None
        and event_log.action_input.action_config is not None
        and isinstance(
            event_log.action_input.action_config.action_trigger, PhraseBasedActionTrigger
        )
    )


def get_openai_chat_messages_from_transcript(
    merged_event_logs: List[EventLog],
    prompt_preamble: str,
) -> List[dict]:
    """Convert transcript events to OpenAI chat messages with proper tool call handling."""
    chat_messages = [{"role": "system", "content": prompt_preamble}]
    
    # Ensure tool calls are paired with the correct response
    tool_call_to_response = {}  # tool_call_id -> ActionFinish
    action_starts_by_id = {}    # tool_call_id -> ActionStart
    
    # First pass: collect all ActionStart and ActionFinish events
    for event_log in merged_event_logs:
        if isinstance(event_log, ActionStart) and not is_phrase_based_action_event_log(event_log):
            if event_log.tool_call_id:
                action_starts_by_id[event_log.tool_call_id] = event_log
        elif isinstance(event_log, ActionFinish) and not is_phrase_based_action_event_log(event_log):
            if event_log.tool_call_id:
                tool_call_to_response[event_log.tool_call_id] = event_log
    
    processed_tool_calls = set()
    
    # Second pass: build chat messages
    i = 0
    while i < len(merged_event_logs):
        event_log = merged_event_logs[i]
        
        if isinstance(event_log, Message):
            if len(event_log.text.strip()) == 0:
                i += 1
                continue
                
            # Check if this bot message is associated with a tool call
            if event_log.sender == Sender.BOT:
                # Look ahead for an ActionStart in the next few events
                associated_tool_calls = []
                j = i + 1
                while j < len(merged_event_logs) and j < i + 5:  # Look ahead
                    next_event = merged_event_logs[j]
                    if isinstance(next_event, ActionStart) and not is_phrase_based_action_event_log(next_event):
                        if (next_event.tool_call_id and 
                            next_event.tool_call_id in tool_call_to_response and
                            next_event.tool_call_id not in processed_tool_calls):
                            associated_tool_calls.append(next_event)
                            processed_tool_calls.add(next_event.tool_call_id)
                            break  # Only associate with the first tool call found
                    elif isinstance(next_event, Message) and next_event.sender == Sender.HUMAN:
                        break
                    j += 1
                
                if associated_tool_calls:
                    message = {
                        "role": "assistant",
                        "content": event_log.to_string(include_sender=False),
                        "tool_calls": [
                            {
                                "id": action.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": action.action_type,
                                    "arguments": action.action_input.params.json(),
                                },
                            }
                            for action in associated_tool_calls
                        ],
                    }
                    chat_messages.append(message)
                    
                    for action in associated_tool_calls:
                        if action.tool_call_id in tool_call_to_response:
                            finish_event = tool_call_to_response[action.tool_call_id]
                            chat_messages.append({
                                "role": "tool",
                                "tool_call_id": action.tool_call_id,
                                "content": finish_event.to_string(include_header=False),
                            })
                else:
                    chat_messages.append({
                        "role": "assistant",
                        "content": event_log.to_string(include_sender=False),
                    })
            else:
                chat_messages.append({
                    "role": "user",
                    "content": event_log.to_string(include_sender=False),
                })
            i += 1
            
        elif isinstance(event_log, ActionStart):
            if (is_phrase_based_action_event_log(event_log) or 
                not event_log.tool_call_id or
                event_log.tool_call_id in processed_tool_calls):
                i += 1
                continue
                
            if event_log.tool_call_id in tool_call_to_response:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": event_log.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": event_log.action_type,
                                "arguments": event_log.action_input.params.json(),
                            },
                        }
                    ],
                }
                chat_messages.append(message)
                processed_tool_calls.add(event_log.tool_call_id)
                
                finish_event = tool_call_to_response[event_log.tool_call_id]
                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": event_log.tool_call_id,
                    "content": finish_event.to_string(include_header=False),
                })
            i += 1
            
        elif isinstance(event_log, ConferenceEvent):
            chat_messages.append(
                {"role": "user", "content": event_log.to_string(include_sender=False)},
            )
            i += 1
        else:
            i += 1
    
    return chat_messages


def merge_event_logs(event_logs: List[EventLog]) -> List[EventLog]:
    """Returns a new list of event logs where consecutive bot messages are merged."""
    new_event_logs: List[EventLog] = []
    idx = 0
    while idx < len(event_logs):
        bot_messages_buffer: List[Message] = []
        current_log = event_logs[idx]
        while isinstance(current_log, Message) and current_log.sender == Sender.BOT:
            bot_messages_buffer.append(current_log)
            idx += 1
            try:
                current_log = event_logs[idx]
            except IndexError:
                break
        if bot_messages_buffer:
            merged_bot_message = deepcopy(bot_messages_buffer[-1])
            merged_bot_message.text = " ".join(event_log.text for event_log in bot_messages_buffer)
            new_event_logs.append(merged_bot_message)
        else:
            new_event_logs.append(current_log)
            idx += 1

    return new_event_logs


def format_openai_chat_messages_from_transcript(
    transcript: Transcript,
    model_name: str,
    functions: Optional[List[Dict]],
    prompt_preamble: str,
) -> List[dict]:
    # merge consecutive bot messages
    merged_event_logs: List[EventLog] = merge_event_logs(event_logs=transcript.event_logs)

    chat_messages: List[Dict[str, Optional[Any]]]
    chat_messages = get_openai_chat_messages_from_transcript(
        merged_event_logs=merged_event_logs,
        prompt_preamble=prompt_preamble,
    )

    context_size = num_tokens_from_messages(
        messages=chat_messages,
        model=model_name,
    ) + num_tokens_from_functions(functions=functions, model=model_name)

    num_removed_messages = 0
    while (
        context_size > get_chat_gpt_max_tokens(model_name) - LLM_AGENT_DEFAULT_MAX_TOKENS - 50
    ):
        if len(chat_messages) <= 1:
            logger.error(f"Prompt is too long to fit in context window, num tokens {context_size}")
            break
        num_removed_messages += 1
        
        # Remove messages more carefully to avoid breaking tool call/response pairs
        # Find the first message that can be safely removed
        removed = False
        for i in range(1, len(chat_messages)):
            msg = chat_messages[i]
            # Skip system message and tool responses
            if msg.get("role") == "system" or msg.get("role") == "tool":
                continue
            # Skip assistant messages with tool calls (need to remove with their responses)
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                continue
            # Safe to remove this message
            chat_messages.pop(i)
            removed = True
            break
            
        if not removed:
            # If we couldn't find a safe message to remove, just remove from index 1
            chat_messages.pop(1)
            
        context_size = num_tokens_from_messages(
            messages=chat_messages,
            model=model_name,
        ) + num_tokens_from_functions(functions=functions, model=model_name)

    if num_removed_messages > 0:
        logger.info(
            "Removed %d messages from prompt to satisfy context limit",
            num_removed_messages,
        )

    return chat_messages


async def openai_get_tokens(
    gen: AsyncGenerator[ChatCompletionChunk, None],
) -> AsyncGenerator[Union[str, FunctionFragment], None]:
    tool_calls = {}
    
    async for event in gen:
        choices = event.choices
        if len(choices) == 0:
            continue
        choice = choices[0]
        if choice.finish_reason:
            if choice.finish_reason == "content_filter":
                logger.warning(
                    "Detected content filter.",
                    extra={"chat_completion_chunk": event.model_dump()},
                )
            break
        delta = choice.delta
        if delta.content is not None:
            token = delta.content
            yield token
        elif delta.tool_calls is not None:
            for tool_call_chunk in delta.tool_calls:
                index = tool_call_chunk.index
                if index not in tool_calls:
                    tool_calls[index] = {
                        "id": "",
                        "name": "",
                        "arguments": "",
                        "name_sent": False
                    }
                
                if tool_call_chunk.id:
                    tool_calls[index]["id"] = tool_call_chunk.id
                
                if tool_call_chunk.function:
                    if tool_call_chunk.function.name:
                        tool_calls[index]["name"] += tool_call_chunk.function.name
                    if tool_call_chunk.function.arguments:
                        tool_calls[index]["arguments"] += tool_call_chunk.function.arguments
                        if index == 0:
                            name_to_send = ""
                            if not tool_calls[index]["name_sent"] and tool_calls[index]["name"]:
                                name_to_send = tool_calls[index]["name"]
                                tool_calls[index]["name_sent"] = True
                            
                            yield FunctionFragment(
                                name=name_to_send,
                                arguments=tool_call_chunk.function.arguments,
                                tool_call_id=tool_calls[index]["id"]
                            )
        elif delta.function_call is not None:
            # Backward compatibility for older models
            yield FunctionFragment(
                name=(delta.function_call.name if delta.function_call.name is not None else ""),
                arguments=(
                    delta.function_call.arguments
                    if delta.function_call.arguments is not None
                    else ""
                ),
            )
