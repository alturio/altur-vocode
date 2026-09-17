from typing import Any, Optional

from vocode.streaming.models.telephony import AlturConfig
from vocode.streaming.telephony.client.abstract_telephony_client import (
    AbstractTelephonyClient,
)
from vocode.streaming.telephony.client.altur_auth import altur_auth_headers
from vocode.streaming.utils.async_requester import AsyncRequestor


class AlturException(Exception):
    pass


class AlturClient(AbstractTelephonyClient):
    def __init__(
        self,
        base_url: str,
        maybe_altur_config: Optional[AlturConfig] = None,
    ):
        self.altur_config = maybe_altur_config or AlturConfig()
        super().__init__(base_url=base_url)

    def get_telephony_config(self):
        return self.altur_config
    
    async def create_call(self, conversation_id: str, to_phone: str, from_phone: str, record: bool = False, digits: Optional[str] = None, telephony_params: Any = None) -> str:
        return conversation_id
    
    async def end_call(self, altur_call_id: str):
        url = f"{self.altur_config.telephony_url}/api/tool/hangup/{altur_call_id}"
        async with AsyncRequestor().get_session().post(
            url,
            headers=altur_auth_headers(url),
            allow_redirects=False,
        ) as response:
            if not 200 <= response.status < 300:
                raise AlturException(f"Failed to end call: {response.status} {response.reason}")
            response = await response.json()
            return response["result"]["success"]
