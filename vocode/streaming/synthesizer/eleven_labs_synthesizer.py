import asyncio
import hashlib
from typing import Optional

from elevenlabs import Voice, VoiceSettings
from elevenlabs.client import AsyncElevenLabs
from loguru import logger

from vocode.streaming.models.audio import AudioEncoding, SamplingRate
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult
from vocode.streaming.synthesizer.audio_cache import AudioCache
from vocode.streaming.utils.create_task import asyncio_create_task

ELEVEN_LABS_BASE_URL = "https://api.elevenlabs.io/v1/"
STREAMED_CHUNK_SIZE = 16000 * 2 // 4  # 1/8 of a second of 16kHz audio with 16-bit samples


class ElevenlabsException(Exception):
    pass


class ElevenLabsSynthesizer(BaseSynthesizer[ElevenLabsSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: ElevenLabsSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        assert synthesizer_config.api_key is not None, "API key must be set"
        assert synthesizer_config.voice_id is not None, "Voice ID must be set"
        self.api_key = synthesizer_config.api_key

        self.elevenlabs_client = AsyncElevenLabs(
            api_key=self.api_key,
        )

        self.model_id = synthesizer_config.model_id
        self.voice_id = synthesizer_config.voice_id
        self.stability = synthesizer_config.stability
        self.similarity_boost = synthesizer_config.similarity_boost
        self.style = synthesizer_config.style
        self.speed = synthesizer_config.speed
        self.use_speaker_boost = synthesizer_config.use_speaker_boost
        self.upsample = None
        self.sample_rate = self.synthesizer_config.sampling_rate

        if self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            match self.synthesizer_config.sampling_rate:
                case SamplingRate.RATE_16000:
                    self.output_format = "pcm_16000"
                case SamplingRate.RATE_22050:
                    self.output_format = "pcm_22050"
                case SamplingRate.RATE_24000:
                    self.output_format = "pcm_24000"
                case SamplingRate.RATE_44100:
                    self.output_format = "pcm_44100"
                case SamplingRate.RATE_48000:
                    self.output_format = "pcm_44100"
                    self.upsample = SamplingRate.RATE_48000.value
                    self.sample_rate = SamplingRate.RATE_44100.value
                case _:
                    raise ValueError(
                        f"Unsupported sampling rate: {self.synthesizer_config.sampling_rate}. Elevenlabs only supports 16000, 22050, 24000, and 44100 Hz."
                    )
        elif self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            self.output_format = "ulaw_8000"
        else:
            raise ValueError(
                f"Unsupported audio encoding: {self.synthesizer_config.audio_encoding}"
            )

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        self.total_chars += len(message.text)
        if self.stability is not None and self.similarity_boost is not None:
            voice = Voice(
                voice_id=self.voice_id,
                settings=VoiceSettings(
                    stability=self.stability,
                    similarity_boost=self.similarity_boost,
                    style=self.style,
                    speed=self.speed,
                    use_speaker_boost=self.use_speaker_boost,
                ),
            )
        else:
            voice = Voice(voice_id=self.voice_id)
        url = (
            ELEVEN_LABS_BASE_URL
            + f"text-to-speech/{self.voice_id}/stream?output_format={self.output_format}"
        )
        headers = {"xi-api-key": self.api_key}
        body = {
            "text": message.text,
            "voice_settings": voice.settings.dict() if voice.settings else None,
        }
        if self.model_id:
            body["model_id"] = self.model_id

        chunk_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        asyncio_create_task(
            self.get_chunks(url, headers, body, chunk_size, chunk_queue),
        )

        return SynthesisResult(
            self.chunk_result_generator_from_queue(chunk_queue),
            lambda seconds: self.get_message_cutoff_from_voice_speed(message, seconds, 150),
        )

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: ElevenLabsSynthesizerConfig):
        return ":".join(
            (
                "eleven_labs",
                str(synthesizer_config.voice_id),
                str(synthesizer_config.model_id),
                str(synthesizer_config.stability),
                str(synthesizer_config.similarity_boost),
                str(synthesizer_config.style),
                str(synthesizer_config.speed),
                str(synthesizer_config.use_speaker_boost),
                synthesizer_config.audio_encoding,
            )
        )

    async def get_chunks(
        self,
        url: str,
        headers: dict,
        body: dict,
        chunk_size: int,
        chunk_queue: asyncio.Queue[Optional[bytes]],
    ):
        audio_buffer = bytearray()
        try:
            async_client = self.async_requestor.get_client()
            stream = await async_client.send(
                async_client.build_request(
                    "POST",
                    url,
                    headers=headers,
                    json=body,
                ),
                stream=True,
            )

            if not stream.is_success:
                error = await stream.aread()
                raise ElevenlabsException(
                    f"ElevenLabs API returned {stream.status_code} status code and the following details: {error.decode('utf-8')}"
                )
            async for chunk in stream.aiter_bytes(chunk_size):
                if self.upsample:
                    chunk = self._resample_chunk(
                        chunk,
                        self.sample_rate,
                        self.upsample,
                    )
                audio_buffer.extend(chunk)
                chunk_queue.put_nowait(chunk)
            
            if self.synthesizer_config.use_cache:
                text = body.get("text", "")
                if text:
                    audio_cache = await AudioCache.safe_create()
                    await audio_cache.set_audio(
                        self.get_voice_identifier(self.synthesizer_config),
                        text.strip(),
                        bytes(audio_buffer)
                    )
        except asyncio.CancelledError:
            pass
        finally:
            chunk_queue.put_nowait(None)  # treated as sentinel
