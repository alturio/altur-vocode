import os
import unittest
from types import SimpleNamespace
from unittest import mock

import aiohttp
import httpx
from aioresponses import aioresponses

from vocode.streaming.action.execute_external_action import (
    ExecuteExternalAction,
    ExecuteExternalActionVocodeActionConfig,
)
from vocode.streaming.models.telephony import AlturConfig
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.telephony.client import altur_client
from vocode.streaming.telephony.client.altur_auth import altur_service_auth
from vocode.streaming.telephony.conversation import altur_phone_conversation


class TestAlturCallbackAuthentication(unittest.IsolatedAsyncioTestCase):
    """Verify authentication across SDK transports without serializing credentials.

    Tests covered:
    - Explicit internal-action authentication survives config serialization.
    - External actions cannot inherit authority through names or URLs.
    - Empty tokens, unexpected origins and redirects fail closed.
    - Altur conversation callbacks and aiohttp hangup authenticate.
    - Generic conversation callbacks remain unauthenticated.
    """

    def setUp(self):
        self.base_url = "http://telephony:8000"
        self.token = "vocode-test-token"
        self.requests = []
        self.redirect_to = None
        real_client = httpx.AsyncClient
        patches = (
            mock.patch.dict(
                os.environ,
                {
                    "DJANGO_HOST": "telephony",
                    "DJANGO_PORT": "8000",
                    "DJANGO_API_TOKEN": self.token,
                },
            ),
            mock.patch.object(
                httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: real_client(
                    **{**kwargs, "transport": httpx.MockTransport(self.handle_request)}
                ),
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def handle_request(self, request):
        self.requests.append(request)
        if self.redirect_to:
            return httpx.Response(307, headers={"Location": self.redirect_to})
        return httpx.Response(200, json={"result": {"success": True}})

    def action_config(self, **overrides):
        return ExecuteExternalActionVocodeActionConfig(
            **{
                "processing_mode": "muted",
                "name": "amd",
                "description": "Test action",
                "url": self.base_url + "/api/tool/amd/cll_test",
                "input_schema": '{"type":"object","properties":{}}',
                "speak_on_send": False,
                "speak_on_receive": True,
                "signature_secret": "c2lnbmF0dXJl",
                "async_execution": False,
                **overrides,
            }
        )

    async def test_action_authentication_is_explicit_and_not_serialized(self):
        """Critical: Only marked actions acquire a credential after deserialization."""
        for internal in (False, True):
            with self.subTest(internal=internal):
                config = self.action_config(altur_auth=internal)
                config = ExecuteExternalActionVocodeActionConfig.parse_raw(config.json())
                action = ExecuteExternalAction(config)
                action_input = action.create_action_input("cll_test", {"answered_by": "human"})
                result = await action.send_external_action_request(action_input)
                self.assertTrue(result.success)
                expected = self.token if internal else None
                self.assertEqual(self.requests[-1].headers.get("Authorization"), expected)
                self.assertNotIn(self.token, action_input.json())
                self.assertNotIn(self.token, config.json())
                self.assertIsNone(config.headers)

    async def test_missing_token_and_wrong_origin_never_send(self):
        """Critical: Missing credentials and authority confusion fail before network I/O."""
        with mock.patch.dict(os.environ, {"DJANGO_API_TOKEN": ""}):
            async with httpx.AsyncClient(auth=altur_service_auth) as client:
                with self.assertRaisesRegex(RuntimeError, "DJANGO_API_TOKEN is not configured"):
                    await client.get(self.base_url + "/api/calls/cll_test/configuration")
        for origin in (
            "http://external.example",
            "https://telephony:8000",
            "http://telephony:8001",
            "http://user:password@telephony:8000",
        ):
            with self.subTest(origin=origin):
                action = ExecuteExternalAction(
                    self.action_config(altur_auth=True, url=origin + "/tool")
                )
                with self.assertRaisesRegex(ValueError, "Unexpected Django service origin"):
                    await action.send_external_action_request(
                        action.create_action_input("cll_test", {})
                    )
        self.assertEqual(self.requests, [])

    async def test_action_redirects_never_replay_or_forward_credentials(self):
        """Critical: Marked actions reject both same-origin and external redirects."""
        action = ExecuteExternalAction(self.action_config(altur_auth=True))
        for origin in (self.base_url, "http://external.example"):
            with self.subTest(origin=origin):
                self.requests.clear()
                self.redirect_to = origin + "/redirected"
                with self.assertRaises(httpx.HTTPStatusError):
                    await action.send_external_action_request(
                        action.create_action_input("cll_test", {})
                    )
                self.assertEqual(len(self.requests), 1)

    async def test_external_credentials_are_preserved_without_service_auth(self):
        """Verification: An arbitrary tool named amd keeps only its configured credential."""
        config = self.action_config(
            url="https://external.example/tool", headers={"Authorization": "external-token"}
        )
        action = ExecuteExternalAction(config)
        result = await action.send_external_action_request(
            action.create_action_input("cll_test", {})
        )
        self.assertTrue(result.success)
        self.assertEqual(self.requests[-1].headers["Authorization"], "external-token")
        self.assertNotIn(self.token, config.json())

    def conversation(self):
        def setup_base(conversation, **kwargs):
            conversation.base_url = kwargs["base_url"]
            conversation.agent = SimpleNamespace(
                get_agent_config=lambda: SimpleNamespace(
                    end_conversation_callback_url=self.base_url
                    + "/api/tool/hangup-with-playback/cll_test?media_uri=test"
                )
            )
            conversation.amd_config = SimpleNamespace(
                callback_url=self.base_url + "/api/tool/amd/cll_test"
            )
            conversation._callback_auth = None

        with (
            mock.patch.object(
                altur_phone_conversation.AbstractPhoneConversation, "__init__", setup_base
            ),
            mock.patch.object(altur_phone_conversation, "AlturOutputDevice"),
        ):
            return altur_phone_conversation.AlturPhoneConversation(
                direction="outbound",
                from_phone="test",
                to_phone="test",
                base_url="test",
                config_manager=None,
                agent_config=None,
                transcriber_config=None,
                synthesizer_config=None,
                amd_config=None,
                altur_call_id="cll_test",
                altur_config=AlturConfig(telephony_url=self.base_url),
                agent_factory=None,
                transcriber_factory=None,
                synthesizer_factory=None,
            )

    async def test_altur_conversation_callbacks_use_runtime_auth(self):
        """Critical: Playback/hangup and keyword AMD use the Altur conversation's auth."""
        conversation = self.conversation()
        await conversation._execute_end_conversation_callback()
        await conversation._send_voicemail_event()
        self.assertEqual(len(self.requests), 2)
        for request in self.requests:
            self.assertEqual(request.headers["Authorization"], self.token)
        self.assertEqual(self.requests[0].url.params["media_uri"], "test")
        self.assertIn(b'"answered_by":"machine"', self.requests[1].content)

    async def test_generic_callbacks_do_not_receive_altur_credentials(self):
        """Verification: Non-Altur callbacks keep their existing unauthenticated transport."""
        conversation = SimpleNamespace(
            _callback_auth=None,
            agent=SimpleNamespace(
                get_agent_config=lambda: SimpleNamespace(
                    end_conversation_callback_url="https://external.example/end"
                )
            ),
            amd_config=SimpleNamespace(callback_url="https://external.example/amd"),
        )
        await StreamingConversation._execute_end_conversation_callback(conversation)
        await StreamingConversation._send_voicemail_event(conversation)
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all("Authorization" not in request.headers for request in self.requests))

    async def test_aiohttp_hangup_auth_is_per_request_and_rejects_redirects(self):
        """Critical: SDK hangup neither changes shared session headers nor follows redirects."""
        client = altur_client.AlturClient("test", AlturConfig(telephony_url=self.base_url))
        url = self.base_url + "/api/tool/hangup/cll_test"
        async with aiohttp.ClientSession() as session:
            with mock.patch.object(altur_client, "AsyncRequestor") as requestor:
                requestor.return_value.get_session.return_value = session
                with aioresponses() as responses:
                    responses.post(url, payload={"result": {"success": True}})
                    self.assertTrue(await client.end_call("cll_test"))
                    call = next(iter(responses.requests.values()))[0]
                    self.assertEqual(call.kwargs["headers"]["Authorization"], self.token)
                    self.assertFalse(call.kwargs["allow_redirects"])
                    self.assertNotIn("Authorization", session.headers)
                for origin in (self.base_url, "http://external.example"):
                    with self.subTest(origin=origin), aioresponses() as responses:
                        responses.post(
                            url, status=307, headers={"Location": origin + "/redirected"}
                        )
                        with self.assertRaises(altur_client.AlturException):
                            await client.end_call("cll_test")
                        self.assertEqual(len(responses.requests), 1)
