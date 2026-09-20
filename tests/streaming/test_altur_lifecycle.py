import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.telephony.constants import (
    ALTUR_AUDIO_ENCODING,
    ALTUR_SAMPLING_RATE,
)
from vocode.streaming.telephony.conversation.altur_phone_conversation import (
    AlturPhoneConversation,
)
from vocode.streaming.utils.provider_lifecycle import (
    ProviderScope,
    complete_provider_request,
    provider_request,
)


class TestAlturLifecycleCancellation(unittest.IsolatedAsyncioTestCase):
    """Exercise SDK startup cancellation without external providers.

    Tests covered:
    - Cancellation during transcriber readiness prevents agent startup.
    - Initial disconnect never starts providers.
    - Forced cleanup attempts every component even when one fails or stalls.
    - Delayed greetings and known worker tasks are cancelled and joined.
    - Managed transcription can acknowledge closure before local cancellation.
    """

    def setUp(self):
        self.config = SimpleNamespace(
            actions=[],
            send_filler_audio=False,
            allowed_idle_time_seconds=120,
            initial_message=None,
        )
        self.agent = SimpleNamespace(
            get_agent_config=lambda: self.config,
            set_interruptible_event_factory=mock.Mock(),
            attach_conversation_state_manager=mock.Mock(),
            attach_speed_manager=mock.Mock(),
            attach_transcript=mock.Mock(),
            start=mock.Mock(),
            terminate=mock.AsyncMock(),
        )
        self.transcriber = SimpleNamespace(
            start=mock.Mock(),
            ready=mock.AsyncMock(),
            terminate=mock.AsyncMock(),
            attach_speed_manager=mock.Mock(),
        )
        self.output = SimpleNamespace(start=mock.Mock(), terminate=mock.AsyncMock())
        self.synthesizer = SimpleNamespace(
            tear_down=mock.AsyncMock(),
            get_synthesizer_config=lambda: SimpleNamespace(
                audio_encoding=ALTUR_AUDIO_ENCODING, sampling_rate=ALTUR_SAMPLING_RATE
            ),
        )
        self.conversation = object.__new__(AlturPhoneConversation)
        StreamingConversation.__init__(
            self.conversation,
            self.output,
            self.transcriber,
            self.agent,
            self.synthesizer,
            amd_config=None,
        )
        self.conversation.to_phone = "to"
        self.conversation.from_phone = "from"
        self.websocket = SimpleNamespace(
            receive=mock.AsyncMock(return_value={"type": "websocket.receive"})
        )

    async def test_cancelled_startup_aborts_every_component(self):
        """Critical: Cancelling partial SDK startup prevents the agent from starting later."""
        entered = asyncio.Event()

        async def wait_for_provider():
            entered.set()
            await asyncio.Event().wait()
            return True

        self.transcriber.ready.side_effect = wait_for_provider
        task = asyncio.create_task(
            self.conversation.attach_ws_and_start(self.websocket)
        )
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.agent.start.assert_not_called()
        for component in (self.agent, self.transcriber, self.output):
            component.terminate.assert_awaited_once()
        self.synthesizer.tear_down.assert_awaited_once()
        self.assertTrue(self.conversation.transcriptions_worker.worker_task.done())
        self.assertTrue(self.conversation.agent_responses_worker.worker_task.done())
        self.assertTrue(self.conversation.synthesis_results_worker.worker_task.done())

    async def test_disconnect_before_start_never_starts_workers(self):
        """Critical: A disconnected initial socket cannot start providers."""
        self.websocket.receive.return_value = {
            "type": "websocket.disconnect",
            "code": 1000,
        }
        await self.conversation.attach_ws_and_start(self.websocket)
        self.transcriber.start.assert_not_called()
        self.agent.start.assert_not_called()
        self.synthesizer.tear_down.assert_awaited_once()

    async def test_managed_abort_waits_for_transcription_completion_before_cancelling(
        self,
    ):
        """Critical: Managed abort permits final transcription metadata and joins its worker before confirming cleanup."""
        scope = ProviderScope(
            deadline=time.monotonic() + 10, providers={"llm", "stt", "tts"}
        )
        self.conversation._provider_scope = self.transcriber._provider_scope = scope
        entered, finish = asyncio.Event(), asyncio.Event()

        async def provider():
            with provider_request(scope) as request:
                entered.set()
                await finish.wait()
                complete_provider_request(request)

        async def terminate():
            self.assertTrue(scope.closed)
            self.assertFalse(self.transcriber.worker_task.cancelled())
            finish.set()
            await self.transcriber.worker_task

        self.transcriber.worker_task = asyncio.create_task(provider())
        self.transcriber.terminate.side_effect = terminate
        await entered.wait()
        await self.conversation.abort()
        self.assertTrue(scope.confirmed)
        self.assertTrue(self.transcriber.worker_task.done())

    async def test_failed_managed_cleanup_cannot_confirm_task_join(self):
        """Error Handling: Provider completion does not conceal a failed local cleanup."""
        scope = ProviderScope(
            deadline=time.monotonic() + 10, providers={"llm", "stt", "tts"}
        )
        self.conversation._provider_scope = scope
        self.synthesizer.tear_down.side_effect = RuntimeError("cleanup failed")
        with self.assertRaisesRegex(
            RuntimeError, "Conversation abort cleanup incomplete"
        ):
            await self.conversation.abort()
        self.assertTrue(scope.closed)
        self.assertFalse(scope.joined)
        self.assertFalse(scope.confirmed)

    async def test_abort_failure_does_not_skip_other_resources(self):
        """Error Handling: One failed cleanup cannot prevent other components from stopping."""
        self.synthesizer.tear_down.side_effect = RuntimeError("provider-secret-canary")
        self.conversation.initial_message_task = asyncio.create_task(
            asyncio.Event().wait()
        )
        with self.assertRaisesRegex(
            RuntimeError, "^Conversation abort cleanup incomplete$"
        ):
            await self.conversation.abort()
        for component in (self.agent, self.transcriber, self.output):
            component.terminate.assert_awaited_once()
        self.assertTrue(self.conversation.initial_message_task.cancelled())

    async def test_abort_does_not_wait_forever_for_provider_cleanup(self):
        """Error Handling: Slow cleanup leaves an explicit failure after the local wait budget."""
        stopped = asyncio.Event()
        wait = asyncio.wait

        async def cleanup():
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def short_wait(tasks, *, timeout):
            self.assertEqual(timeout, 5.0)
            return await wait(tasks, timeout=0.01)

        self.synthesizer.tear_down.side_effect = cleanup
        with mock.patch(
            "vocode.streaming.streaming_conversation.asyncio.wait",
            side_effect=short_wait,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "^Conversation abort cleanup incomplete$"
            ):
                await self.conversation.abort()
        await asyncio.wait_for(stopped.wait(), 1)
        self.transcriber.terminate.assert_awaited_once()

    async def test_cancellation_survives_cleanup_failure(self):
        """Critical: Cleanup failure cannot turn shutdown cancellation into a successful return."""
        entered = asyncio.Event()

        async def receive():
            entered.set()
            await asyncio.Event().wait()

        self.websocket.receive.side_effect = receive
        self.synthesizer.tear_down.side_effect = RuntimeError("provider-secret-canary")
        task = asyncio.create_task(
            self.conversation.attach_ws_and_start(self.websocket)
        )
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.transcriber.terminate.assert_awaited_once()
