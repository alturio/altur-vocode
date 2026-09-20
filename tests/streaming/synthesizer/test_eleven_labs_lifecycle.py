import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.synthesizer.eleven_labs_synthesizer import (
    ElevenlabsException,
    ElevenLabsSynthesizer,
)
from vocode.streaming.utils.provider_lifecycle import ProviderScope


class RecordingStream(httpx.AsyncByteStream):
    def __init__(self, *, blocked=False, close_error=False):
        self.blocked = blocked
        self.close_error = close_error
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.started.set()
        yield b"audio"
        if self.blocked:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True
        if self.close_error:
            raise RuntimeError("provider-secret-canary")


class TestElevenLabsProducerLifecycle(unittest.IsolatedAsyncioTestCase):
    """Own synthesis producers and their individual HTTP responses.

    Tests covered:
    - Normal completion and cancellation close response streams
    - Teardown joins producers without closing other calls' shared HTTP client
    - Closed synthesizers reject late work and cleanup failures remain visible
    - Provider response bodies are not included in errors
    - Only fully consumed provider responses supply completion evidence
    """

    async def asyncSetUp(self):
        self.stream = RecordingStream()
        self.status = 200
        self.requests = []

        def handle(request):
            self.requests.append(request)
            return httpx.Response(self.status, stream=self.stream)

        self.client = await self.enterAsyncContext(
            httpx.AsyncClient(transport=httpx.MockTransport(handle))
        )
        self.enterContext(
            patch(
                "vocode.streaming.synthesizer.base_synthesizer.AsyncRequestor",
                return_value=SimpleNamespace(get_client=lambda: self.client),
            )
        )
        self.synthesizer = ElevenLabsSynthesizer(
            ElevenLabsSynthesizerConfig.from_altur_output_device(
                api_key="test",
                voice_id="test",
                use_cache=False,
            )
        )

    async def speech(self):
        return await self.synthesizer.create_speech_uncached(
            BaseMessage(text="Hello"), 1
        )

    async def test_normal_completion_closes_response(self):
        """Basic: Finite synthesis drains normally and releases its individual HTTP response."""
        result = await self.speech()
        chunks = [chunk.chunk async for chunk in result.chunk_generator]
        self.assertEqual(b"".join(chunks), b"audio")
        await self.synthesizer.tear_down()
        self.assertTrue(self.stream.closed)
        self.assertFalse(self.client.is_closed)
        self.assertFalse(self.synthesizer._chunk_tasks)

    async def test_managed_response_completion_is_distinct_from_transport_cleanup(self):
        """Verification: Fully consumed synthesis confirms its request without depending on ambient task context."""
        owner = ProviderScope(deadline=time.monotonic() + 10)
        self.synthesizer._provider_scope = owner
        result = await self.speech()
        self.assertTrue([chunk async for chunk in result.chunk_generator])
        await self.synthesizer.tear_down()
        self.assertEqual(owner.pending, 0)
        self.assertFalse(owner.uncertain)

    async def test_managed_cancellation_retains_unknown_provider_outcome(self):
        """Critical: A cancelled response does not certify remote synthesis completion even after local closure."""
        owner = ProviderScope(deadline=time.monotonic() + 10)
        self.synthesizer._provider_scope = owner
        self.stream.blocked = True
        await self.speech()
        await asyncio.wait_for(self.stream.started.wait(), 1)
        await self.synthesizer.tear_down()
        self.assertTrue(self.stream.closed)
        self.assertEqual(owner.pending, 0)
        self.assertTrue(owner.uncertain)

    async def test_teardown_cancels_and_joins_blocked_producers(self):
        """Critical: Teardown closes in-flight synthesis without shutting down the shared client."""
        self.stream.blocked = True
        await self.speech()
        await asyncio.wait_for(self.stream.started.wait(), 1)
        tasks = tuple(self.synthesizer._chunk_tasks)
        await asyncio.wait_for(self.synthesizer.tear_down(), 1)
        self.assertTrue(self.stream.closed)
        self.assertTrue(all(task.done() for task in tasks))
        self.assertFalse(self.synthesizer._chunk_tasks)
        self.assertFalse(self.client.is_closed)
        await self.synthesizer.tear_down()
        with self.assertRaisesRegex(RuntimeError, "synthesizer is closed"):
            await self.speech()
        self.assertEqual(len(self.requests), 1)

    async def test_teardown_cannot_close_another_calls_stream(self):
        """Critical: Concurrent calls sharing the HTTP pool retain independent producer lifetimes."""
        first = self.stream
        first.blocked = True
        await self.speech()
        await asyncio.wait_for(first.started.wait(), 1)
        self.stream = second = RecordingStream(blocked=True)
        other = ElevenLabsSynthesizer(self.synthesizer.synthesizer_config)
        await other.create_speech_uncached(BaseMessage(text="Other call"), 1)
        await asyncio.wait_for(second.started.wait(), 1)
        try:
            await self.synthesizer.tear_down()
            self.assertTrue(first.closed)
            self.assertFalse(second.closed)
            self.assertTrue(any(not task.done() for task in other._chunk_tasks))
            self.assertFalse(self.client.is_closed)
        finally:
            await other.tear_down()
        self.assertTrue(second.closed)

    async def test_teardown_before_producer_start_prevents_request(self):
        """Race Condition: A queued producer cannot issue HTTP after teardown has completed."""
        await self.speech()
        await self.synthesizer.tear_down()
        self.assertEqual(self.requests, [])

    async def test_failed_response_close_does_not_report_cleanup_success(self):
        """Error Handling: Response close failures remain visible on cleanup and subsequent retries."""
        self.stream.blocked = True
        self.stream.close_error = True
        await self.speech()
        await asyncio.wait_for(self.stream.started.wait(), 1)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "ElevenLabs cleanup incomplete"):
                await self.synthesizer.tear_down()
        self.assertFalse(self.client.is_closed)
        self.assertTrue(all(task.done() for task in self.synthesizer._chunk_tasks))

    async def test_http_error_does_not_read_or_expose_response_body(self):
        """Critical: Failed provider responses are closed without consuming their arbitrary bodies."""
        self.status = 503
        queue = asyncio.Queue()
        with self.assertRaisesRegex(ElevenlabsException, r"request failed \(503\)"):
            await self.synthesizer.get_chunks("https://example.test", {}, {}, 1, queue)
        self.assertFalse(self.stream.started.is_set())
        self.assertTrue(self.stream.closed)
        self.assertIsNone(queue.get_nowait())
