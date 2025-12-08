import asyncio
import time
from abc import abstractmethod

from vocode.streaming.constants import PER_CHUNK_ALLOWANCE_SECONDS
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.output_device.audio_chunk import ChunkState
from vocode.streaming.utils import get_chunk_size_per_second


class RateLimitInterruptionsOutputDevice(AbstractOutputDevice):
    """Output device that works by rate limiting the chunks sent to the output. For interrupts to work properly,
    the next chunk of audio can only be sent after the last chunk is played, so we send
    a chunk of x seconds only after x seconds have passed since the last chunk was sent."""

    def __init__(
        self,
        sampling_rate: int,
        audio_encoding: AudioEncoding,
        per_chunk_allowance_seconds: float = PER_CHUNK_ALLOWANCE_SECONDS,
        call_id: str | None = None,
    ):
        super().__init__(sampling_rate, audio_encoding)
        self.per_chunk_allowance_seconds = per_chunk_allowance_seconds
        self.call_id = call_id
        self._is_processing = False

    async def _run_loop(self):
        while True:
            start_time = time.time()
            try:
                item = await self._input_queue.get()
            except asyncio.CancelledError:
                return

            self._is_processing = True
            self.interruptible_event = item
            audio_chunk = item.payload

            if item.is_interrupted():
                audio_chunk.on_interrupt()
                audio_chunk.state = ChunkState.INTERRUPTED
                self._is_processing = False
                continue

            speech_length_seconds = (len(audio_chunk.data)) / get_chunk_size_per_second(
                self.audio_encoding,
                self.sampling_rate,
            )
            await self.play(audio_chunk.data, self.call_id)
            audio_chunk.on_play()
            audio_chunk.state = ChunkState.PLAYED
            end_time = time.time()
            await asyncio.sleep(
                max(
                    speech_length_seconds
                    - (end_time - start_time)
                    - self.per_chunk_allowance_seconds,
                    0,
                ),
            )
            self.interruptible_event.is_interruptible = False
            self._is_processing = False

    async def wait_for_drain(self, timeout: float = 30.0) -> bool:
        """Wait for the output queue to drain and current processing to complete.
        
        Returns True if drained successfully, False if timeout.
        """
        start_time = time.time()
        while True:
            if self._input_queue.empty() and not self._is_processing:
                return True
            if time.time() - start_time > timeout:
                return False
            await asyncio.sleep(0.1)

    @abstractmethod
    async def play(self, chunk: bytes, call_id: str | None = None):
        """Sends an audio chunk to immediate playback"""
        pass

    def interrupt(self):
        """
        For conversations that use rate-limiting playback as above,
        no custom logic is needed on interrupt, because to end synthesis, all we need to do is stop sending chunks.
        """
        pass
