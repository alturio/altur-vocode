import json
import asyncio
from typing import Any, Dict, Optional, Type

from pydantic.v1 import BaseModel

from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.action.external_actions_requester import (
    ExternalActionResponse,
    ExternalActionsRequester,
)
from vocode.streaming.models.actions import ActionConfig as VocodeActionConfig
from vocode.streaming.models.actions import ActionInput, ActionOutput, ExternalActionProcessingMode
from vocode.streaming.models.message import BaseMessage


class ExecuteExternalActionVocodeActionConfig(
    VocodeActionConfig, type="action_external"  # type: ignore
):
    processing_mode: ExternalActionProcessingMode
    name: str
    description: str
    url: str
    input_schema: str
    speak_on_send: bool
    speak_on_receive: bool
    signature_secret: str
    async_execution: bool
    headers: Optional[Dict[str, str]] = None
    wrap_arguments: bool = True
    

class ExecuteExternalActionParameters(BaseModel):
    payload: Dict[str, Any]


class ExecuteExternalActionResponse(BaseModel):
    success: bool
    result: Optional[dict]


class ExecuteExternalAction(
    BaseAction[
        ExecuteExternalActionVocodeActionConfig,
        ExecuteExternalActionParameters,
        ExecuteExternalActionResponse,
    ]
):
    parameters_type: Type[ExecuteExternalActionParameters] = ExecuteExternalActionParameters
    response_type: Type[ExecuteExternalActionResponse] = ExecuteExternalActionResponse

    def __init__(
        self,
        action_config: ExecuteExternalActionVocodeActionConfig,
    ):
        self.description = action_config.description
        self.headers = action_config.headers or {}
        self.wrap_arguments = action_config.wrap_arguments
        super().__init__(
            action_config,
            quiet=not action_config.speak_on_receive,
            should_respond="always" if action_config.speak_on_send else "never",
            is_interruptible=False,
        )
        self.external_actions_requester = ExternalActionsRequester(url=action_config.url)

    def _user_message_param_info(self):
        return {
            "type": "string",
            "description": (
                "A message to reply to the user with BEFORE we make the function call."
                "\nEssentially a live response informing them that the function is "
                "about to happen."
            ),
        }

    def create_action_params(self, params: Dict[str, Any]) -> ExecuteExternalActionParameters:
        return ExecuteExternalActionParameters(payload=params)

    def get_function_name(self) -> str:
        return self.action_config.name

    def get_parameters_schema(self) -> Dict[str, Any]:
        return json.loads(self.action_config.input_schema)

    def _process_parameters_by_location(
        self, payload: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Process parameters by their location (path, query, or body).
        
        Args:
            payload: The full parameter payload from the LLM
            
        Returns:
            Tuple of (path_params, query_params, body_params)
        """
        input_schema = json.loads(self.action_config.input_schema)
        parameter_locations = input_schema.get("x-parameter-locations", {})
        
        path_params = {}
        query_params = {}
        body_params = {}
        
        for param_name, param_value in payload.items():
            location = parameter_locations.get(param_name, "body")
            
            if location == "path":
                path_params[param_name] = param_value
            elif location == "query":
                query_params[param_name] = param_value
            else:  # Default to body
                body_params[param_name] = param_value
        
        return path_params, query_params, body_params

    def _build_request_url(
        self, path_params: Dict[str, Any], query_params: Dict[str, Any]
    ) -> str:
        """Build the final request URL with path and query parameters.
        
        Args:
            path_params: Parameters to replace in URL path template
            query_params: Parameters to append as query string
            
        Returns:
            Final URL with path parameters replaced and query params appended
        """
        url = self.action_config.url
        
        if path_params:
            try:
                url = url.format(**path_params)
            except KeyError as e:
                raise ValueError(f"Missing required path parameter: {e}")
        
        if query_params:
            query_string_params = {k: str(v) for k, v in query_params.items()}
            from urllib.parse import urlencode
            query_string = urlencode(query_string_params)
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{query_string}"
        
        return url

    async def send_external_action_request(
        self, action_input: ActionInput[ExecuteExternalActionParameters]
    ) -> ExternalActionResponse:
        path_params, query_params, body_params = self._process_parameters_by_location(
            action_input.params.payload
        )
        
        request_url = self._build_request_url(path_params, query_params)
        
        if self.action_config.async_execution:
            asyncio.create_task(
                self.external_actions_requester.send_request(
                    payload=body_params,
                    signature_secret=self.action_config.signature_secret,
                    additional_headers=self.headers,
                    url=request_url,
                    wrap_arguments=self.wrap_arguments,
                )
            )
            return ExternalActionResponse(
                result={"info": "success"}, success=True
            )
        else:
            return await self.external_actions_requester.send_request(
                payload=body_params,
                signature_secret=self.action_config.signature_secret,
                additional_headers=self.headers,
                url=request_url,
                wrap_arguments=self.wrap_arguments,
            )

    async def run(
        self, action_input: ActionInput[ExecuteExternalActionParameters]
    ) -> ActionOutput[ExecuteExternalActionResponse]:
        # TODO: this interruption handling needs to be refactored / DRYd
        if self.should_respond and action_input.user_message_tracker is not None:
            await action_input.user_message_tracker.wait()

        self.conversation_state_manager.mute_agent()
        response = await self.send_external_action_request(action_input)
        self.conversation_state_manager.unmute_agent()

        # TODO (EA): pass specific context based on error
        return ActionOutput(
            action_type=action_input.action_config.type,
            response=ExecuteExternalActionResponse(
                result=response.result if response else None, success=response.success
            ),
            canned_response=(
                BaseMessage(text=response.agent_message)
                if response and response.agent_message
                else None
            ),
        )
