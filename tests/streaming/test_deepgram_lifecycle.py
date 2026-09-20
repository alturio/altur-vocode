import asyncio
import json
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.utils.provider_lifecycle import (
    ProviderScope,
    current_provider_scope,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["valid", "missing", "invalid", "unsolicited"])
async def test_shutdown_requires_final_metadata_after_close_stream(mocker, reply):
    """Critical: Only a final Deepgram summary for the closing stream confirms remote processing completion."""
    owner = ProviderScope(deadline=time.monotonic() + 10)
    token = current_provider_scope.set(owner)
    try:
        transcriber = DeepgramTranscriber(
            DeepgramTranscriberConfig.from_altur_input_device(api_key="test-only")
        )
    finally:
        current_provider_scope.reset(token)
    messages = asyncio.Queue()
    connected, read, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sent = []
    metadata = {
        "type": "Metadata",
        "request_id": str(uuid4()),
        "duration": 0.0,
        "channels": 0,
    }

    class Socket:
        async def send(self, data):
            sent.append(data)
            if json.loads(data) == {"type": "CloseStream"} and reply in (
                "valid",
                "invalid",
            ):
                await messages.put(
                    json.dumps(
                        metadata if reply == "valid" else metadata | {"duration": True}
                    )
                )

        async def recv(self):
            result = await messages.get()
            read.set()
            return result

    @asynccontextmanager
    async def connect(*args, **kwargs):
        assert kwargs["close_timeout"] == 1
        connected.set()
        try:
            yield Socket()
        finally:
            closed.set()

    mocker.patch(
        "vocode.streaming.transcriber.deepgram_transcriber.websockets.connect", connect
    )
    task = transcriber.start()
    try:
        await asyncio.wait_for(connected.wait(), 1)
        if reply == "unsolicited":
            await messages.put(json.dumps(metadata))
            await asyncio.wait_for(read.wait(), 1)
            assert not transcriber._provider_finished.is_set()
        if reply == "valid":
            await transcriber.terminate()
        else:
            with pytest.raises(TimeoutError):
                await transcriber.terminate()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
        assert owner.pending == 0
        assert owner.uncertain == (reply != "valid")
        assert closed.is_set()
        assert sent == [json.dumps({"type": "CloseStream"})]
        await transcriber.terminate()
        assert owner.uncertain == (reply != "valid")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_transport_cancellation_is_not_a_provider_completion_receipt(mocker):
    """Critical: Closing the local socket without final metadata preserves unknown remote work."""
    owner = ProviderScope(deadline=time.monotonic() + 10)
    token = current_provider_scope.set(owner)
    try:
        transcriber = DeepgramTranscriber(
            DeepgramTranscriberConfig.from_altur_input_device(api_key="test-only")
        )
    finally:
        current_provider_scope.reset(token)
    connected, closed = asyncio.Event(), asyncio.Event()

    class Socket:
        async def recv(self):
            await asyncio.Event().wait()

    @asynccontextmanager
    async def connect(*args, **kwargs):
        connected.set()
        try:
            yield Socket()
        finally:
            closed.set()

    mocker.patch(
        "vocode.streaming.transcriber.deepgram_transcriber.websockets.connect", connect
    )
    task = transcriber.start()
    await asyncio.wait_for(connected.wait(), 1)
    task.cancel()
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
    assert closed.is_set() and owner.pending == 0 and owner.uncertain
