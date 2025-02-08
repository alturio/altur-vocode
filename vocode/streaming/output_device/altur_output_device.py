import json
import base64
from typing import Optional

from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from vocode.streaming.output_device.blocking_speaker_output import BlockingSpeakerOutput
from vocode.streaming.output_device.rate_limit_interruptions_output_device import (
    RateLimitInterruptionsOutputDevice,
)
from vocode.streaming.telephony.constants import (
    ALTUR_AUDIO_ENCODING,
    ALTUR_CHUNK_SIZE,
    ALTUR_SAMPLING_RATE,
    MULAW_SILENCE_BYTE,
)


class AlturOutputDevice(RateLimitInterruptionsOutputDevice):
    def __init__(
        self,
        call_id: str,
        ws: Optional[WebSocket] = None,
        output_to_speaker: bool = False,
    ):
        super().__init__(
            call_id=call_id,
            sampling_rate=ALTUR_SAMPLING_RATE,
            audio_encoding=ALTUR_AUDIO_ENCODING,
        )
        self.ws = ws
        self.output_to_speaker = output_to_speaker
        if output_to_speaker:
            self.output_speaker = BlockingSpeakerOutput.from_default_device(
                sampling_rate=ALTUR_SAMPLING_RATE, blocksize=ALTUR_CHUNK_SIZE // 2
            )

    async def play(self, chunk: bytes, call_id: str):
        if self.output_to_speaker:
            self.output_speaker.consume_nonblocking(chunk)
        for i in range(0, len(chunk), ALTUR_CHUNK_SIZE):
            subchunk = chunk[i : i + ALTUR_CHUNK_SIZE]

            if len(subchunk) < ALTUR_CHUNK_SIZE:
                subchunk += MULAW_SILENCE_BYTE * (ALTUR_CHUNK_SIZE - len(subchunk))

            if self.ws and self.ws.application_state != WebSocketState.DISCONNECTED:
                await self.ws.send_text(
                    json.dumps(
                        {
                            "call_id": call_id,
                            "payload": base64.b64encode(subchunk).decode(),
                        }
                    )
                )
