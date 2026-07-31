import json
import os
from typing import AsyncGenerator

import httpx
from google import genai
from google.genai import types

from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.default_factory import DefaultActionFactory
from vocode.streaming.agent.base_agent import GeneratedResponse, RespondAgent, StreamedResponse
from vocode.streaming.agent.streaming_utils import collate_response_async, stream_response_async
from vocode.streaming.models.actions import FunctionCallActionTrigger, FunctionFragment
from vocode.streaming.models.agent import GeminiAgentConfig
from vocode.streaming.models.message import BaseMessage, LLMToken
from vocode.streaming.utils.date_utils import inject_parsed_dates


class GeminiAgent(RespondAgent[GeminiAgentConfig]):
    """Vocode streaming agent backed by the Gemini Developer API."""

    def __init__(
        self,
        agent_config: GeminiAgentConfig,
        action_factory: AbstractActionFactory = DefaultActionFactory(),
        **kwargs,
    ):
        super().__init__(
            agent_config=agent_config,
            action_factory=action_factory,
            **kwargs,
        )
        api_key = (
            agent_config.google_api_key
            or os.getenv("FASTAPI_GOOGLE_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be set for Gemini models")
        self._httpx_client = httpx.AsyncClient()
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(httpx_async_client=self._httpx_client),
        )

    def get_functions(self):
        assert self.agent_config.actions
        if not self.action_factory:
            return None
        return [
            self.action_factory.create_action(action_config).get_openai_function()
            for action_config in self.agent_config.actions
            if isinstance(action_config.action_trigger, FunctionCallActionTrigger)
        ]

    def _generation_config(self, *, include_tools: bool) -> types.GenerateContentConfig:
        kwargs = {
            "system_instruction": self.agent_config.prompt_preamble,
            "max_output_tokens": self.agent_config.max_tokens,
        }
        if self.agent_config.temperature is not None:
            kwargs["temperature"] = self.agent_config.temperature
        if self.agent_config.thinking_level is not None:
            kwargs["thinking_config"] = {
                "thinking_level": self.agent_config.thinking_level,
            }
        if include_tools and self.functions:
            declarations = [
                types.FunctionDeclaration(
                    name=function["name"],
                    description=function.get("description"),
                    parameters_json_schema=function["parameters"],
                )
                for function in self.functions
            ]
            kwargs["tools"] = [types.Tool(function_declarations=declarations)]
        return types.GenerateContentConfig(**kwargs)

    async def _token_generator(
        self, stream
    ) -> AsyncGenerator[str | FunctionFragment, None]:
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
            for call in chunk.function_calls or []:
                yield FunctionFragment(
                    name=call.name,
                    arguments=json.dumps(call.args or {}),
                    tool_call_id=getattr(call, "id", None),
                )

    async def generate_response(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
        bot_was_in_medias_res: bool = False,
        is_tool_response: bool = False,
    ) -> AsyncGenerator[GeneratedResponse, None]:
        if not self.transcript:
            raise ValueError("A transcript is not attached to the agent")
        if self.agent_config.date_parsing_enabled and not is_tool_response:
            human_input = inject_parsed_dates(
                human_input,
                languages=self.agent_config.date_parsing_languages,
                timezone=self.agent_config.date_parsing_timezone,
            )

        contents = (
            self.transcript.to_string(
                include_timestamps=False,
                mark_human_backchannels_with_brackets=True,
            )
            + "\nBOT:"
        )
        stream = await self._client.aio.models.generate_content_stream(
            model=self.agent_config.model_name,
            contents=contents,
            config=self._generation_config(include_tools=not is_tool_response),
        )
        response_generator = (
            stream_response_async
            if self.conversation_state_manager.using_input_streaming_synthesizer()
            else collate_response_async
        )
        async for message in response_generator(
            conversation_id=conversation_id,
            gen=self._token_generator(stream),
            get_functions=True,
        ):
            response_class = (
                StreamedResponse
                if self.conversation_state_manager.using_input_streaming_synthesizer()
                else GeneratedResponse
            )
            message_type = (
                LLMToken
                if self.conversation_state_manager.using_input_streaming_synthesizer()
                else BaseMessage
            )
            yield response_class(
                message=message_type(text=message)
                if isinstance(message, str)
                else message,
                is_interruptible=True,
            )

    async def terminate(self):
        await self._httpx_client.aclose()
        return await super().terminate()
