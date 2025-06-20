from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic.v1 import validator

from vocode.streaming.models.client_backend import OutputAudioConfig
from vocode.streaming.output_device.abstract_output_device import AbstractOutputDevice
from vocode.streaming.telephony.constants import (
    TWILIO_AUDIO_ENCODING,
    TWILIO_SAMPLING_RATE,
    VONAGE_AUDIO_ENCODING,
    VONAGE_SAMPLING_RATE,
    ALTUR_SAMPLING_RATE,
    ALTUR_AUDIO_ENCODING,
)

from .audio import AudioEncoding, SamplingRate
from .model import BaseModel, TypedModel


class SynthesizerType(str, Enum):
    BASE = "synthesizer_base"
    AZURE = "synthesizer_azure"
    GOOGLE = "synthesizer_google"
    ELEVEN_LABS = "synthesizer_eleven_labs"
    RIME = "synthesizer_rime"
    PLAY_HT = "synthesizer_play_ht"
    GTTS = "synthesizer_gtts"
    STREAM_ELEMENTS = "synthesizer_stream_elements"
    COQUI_TTS = "synthesizer_coqui_tts"
    COQUI = "synthesizer_coqui"
    BARK = "synthesizer_bark"
    POLLY = "synthesizer_polly"
    CARTESIA = "synthesizer_cartesia"


class SentimentConfig(BaseModel):
    emotions: List[str] = ["angry", "friendly", "sad", "whispering"]

    @validator("emotions")
    def emotions_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError("must have at least one emotion")
        return v


class SynthesizerConfig(TypedModel, type=SynthesizerType.BASE.value):  # type: ignore
    sampling_rate: int
    audio_encoding: AudioEncoding
    should_encode_as_wav: bool = False
    sentiment_config: Optional[SentimentConfig] = None
    use_cache: bool = True

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_output_device(cls, output_device: AbstractOutputDevice, **kwargs):
        return cls(
            sampling_rate=output_device.sampling_rate,
            audio_encoding=output_device.audio_encoding,
            **kwargs
        )

    @classmethod
    def from_twilio_output_device(cls, **kwargs):
        return cls(
            sampling_rate=TWILIO_SAMPLING_RATE, audio_encoding=TWILIO_AUDIO_ENCODING, **kwargs
        )

    @classmethod
    def from_vonage_output_device(cls, **kwargs):
        return cls(
            sampling_rate=VONAGE_SAMPLING_RATE, audio_encoding=VONAGE_AUDIO_ENCODING, **kwargs
        )
    
    @classmethod
    def from_altur_output_device(cls, **kwargs):
        return cls(
            sampling_rate=ALTUR_SAMPLING_RATE, audio_encoding=ALTUR_AUDIO_ENCODING, **kwargs
        )

    @classmethod
    def from_output_audio_config(cls, output_audio_config: OutputAudioConfig, **kwargs):
        return cls(
            sampling_rate=output_audio_config.sampling_rate,
            audio_encoding=output_audio_config.audio_encoding,
            **kwargs
        )


AZURE_SYNTHESIZER_DEFAULT_VOICE_NAME = "en-US-SteffanNeural"
AZURE_SYNTHESIZER_DEFAULT_PITCH = 0
AZURE_SYNTHESIZER_DEFAULT_RATE = 15


class AzureSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.AZURE.value):  # type: ignore
    voice_name: str = AZURE_SYNTHESIZER_DEFAULT_VOICE_NAME
    pitch: int = AZURE_SYNTHESIZER_DEFAULT_PITCH
    rate: int = AZURE_SYNTHESIZER_DEFAULT_RATE
    language_code: str = "en-US"


DEFAULT_GOOGLE_LANGUAGE_CODE = "en-US"
DEFAULT_GOOGLE_VOICE_NAME = "en-US-Neural2-I"
DEFAULT_GOOGLE_PITCH = 0
DEFAULT_GOOGLE_SPEAKING_RATE = 1.2


class GoogleSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.GOOGLE.value):  # type: ignore
    language_code: str = DEFAULT_GOOGLE_LANGUAGE_CODE
    voice_name: str = DEFAULT_GOOGLE_VOICE_NAME
    pitch: float = DEFAULT_GOOGLE_PITCH
    speaking_rate: float = DEFAULT_GOOGLE_SPEAKING_RATE


ELEVEN_LABS_ADAM_VOICE_ID = "pNInz6obpgDQGcFmaJgB"


class ElevenLabsSynthesizerConfig(
    SynthesizerConfig, type=SynthesizerType.ELEVEN_LABS.value  # type: ignore
):
    api_key: Optional[str] = None
    model_id: Optional[str]
    voice_id: Optional[str] = ELEVEN_LABS_ADAM_VOICE_ID
    language: Optional[str] = "es"
    stability: Optional[float]
    similarity_boost: Optional[float]
    style: Optional[float] = 0
    use_speaker_boost: Optional[bool] = True
    speed: Optional[float] = 1.0
    experimental_streaming: bool = False
    experimental_websocket: bool = False
    backchannel_amplitude_factor: float = 0.5

    @validator("voice_id")
    def set_name(cls, voice_id):
        return voice_id or ELEVEN_LABS_ADAM_VOICE_ID

    @validator("similarity_boost", always=True)
    def stability_and_similarity_boost_check(cls, similarity_boost, values):
        stability = values.get("stability")
        if (stability is None) != (similarity_boost is None):
            raise ValueError("Both stability and similarity_boost must be set or not set.")
        return similarity_boost

    @validator("speed")
    def speed_check(cls, speed):
        if speed is not None and not (0.7 <= speed <= 1.2):
            raise ValueError("speed must be between 0.7 and 1.2.")
        return speed

    @validator("backchannel_amplitude_factor")
    def backchannel_amplitude_factor_check(cls, backchannel_amplitude_factor):
        if backchannel_amplitude_factor is not None and not (0 < backchannel_amplitude_factor <= 1):
            raise ValueError(
                "backchannel_amplitude_factor must be between 0 (not inclusive) and 1."
            )
        return backchannel_amplitude_factor


RIME_DEFAULT_BASE_URL = "https://users.rime.ai/v1/rime-tts"
RIME_DEFAULT_MODEL_ID = None
RIME_DEFAULT_SPEAKER = "young_male_unmarked-1"
RIME_DEFAULT_SPEED_ALPHA = 1.0
RIME_DEFAULT_SAMPLE_RATE = SamplingRate.RATE_22050
RIME_DEFAULT_REDUCE_LATENCY = False
RimeModelId = Literal["mist", "v1"]


class RimeSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.RIME.value):  # type: ignore
    base_url: str = RIME_DEFAULT_BASE_URL
    model_id: Optional[Literal[RimeModelId]] = RIME_DEFAULT_MODEL_ID
    speaker: str = RIME_DEFAULT_SPEAKER
    speed_alpha: Optional[float] = RIME_DEFAULT_SPEED_ALPHA
    sampling_rate: int = RIME_DEFAULT_SAMPLE_RATE
    reduce_latency: Optional[bool] = RIME_DEFAULT_REDUCE_LATENCY


COQUI_DEFAULT_SPEAKER_ID = "ebe2db86-62a6-49a1-907a-9a1360d4416e"


class CoquiSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.COQUI.value):  # type: ignore
    api_key: Optional[str] = None
    voice_id: Optional[str] = COQUI_DEFAULT_SPEAKER_ID
    voice_prompt: Optional[str] = None
    use_xtts: Optional[bool] = True

    @validator("voice_id", always=True)
    def override_voice_id_with_prompt(cls, voice_id, values):
        if values.get("voice_prompt"):
            return None
        return voice_id or COQUI_DEFAULT_SPEAKER_ID


PlayHtVoiceVersionType = Literal["1", "2"]


class PlayHtSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.PLAY_HT.value):  # type: ignore
    voice_id: str
    api_key: Optional[str] = None
    user_id: Optional[str] = None
    speed: Optional[float] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None
    quality: Optional[str] = None
    experimental_streaming: bool = False
    version: Literal[PlayHtVoiceVersionType] = "2"
    top_p: Optional[float] = None
    text_guidance: Optional[float] = None
    voice_guidance: Optional[float] = None
    on_prem: bool = False
    on_prem_provider: Literal["aws", "gcp"] = "gcp"
    experimental_remove_silence: bool = False


class CoquiTTSSynthesizerConfig(
    SynthesizerConfig, type=SynthesizerType.COQUI_TTS.value  # type: ignore
):
    tts_kwargs: dict = {}
    speaker: Optional[str] = None
    language: Optional[str] = None


class GTTSSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.GTTS.value):  # type: ignore
    pass


STREAM_ELEMENTS_SYNTHESIZER_DEFAULT_VOICE = "Brian"


class StreamElementsSynthesizerConfig(
    SynthesizerConfig, type=SynthesizerType.STREAM_ELEMENTS.value  # type: ignore
):
    voice: str = STREAM_ELEMENTS_SYNTHESIZER_DEFAULT_VOICE


class BarkSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.BARK.value):  # type: ignore
    preload_kwargs: Dict[str, Any] = {}
    generate_kwargs: Dict[str, Any] = {}


DEFAULT_POLLY_LANGUAGE_CODE = "en-US"
DEFAULT_POLLY_VOICE_ID = "Matthew"
DEFAULT_POLLY_SAMPLING_RATE = SamplingRate.RATE_16000.value


class PollySynthesizerConfig(SynthesizerConfig, type=SynthesizerType.POLLY.value):  # type: ignore
    language_code: str = DEFAULT_POLLY_LANGUAGE_CODE
    voice_id: str = DEFAULT_POLLY_VOICE_ID
    sampling_rate: int = DEFAULT_POLLY_SAMPLING_RATE


DEFAULT_CARTESIA_MODEL_ID = "sonic-english"
DEFAULT_CARTESIA_VOICE_ID = "f9836c6e-a0bd-460e-9d3c-f7299fa60f94"


class CartesiaVoiceControls(BaseModel):
    """See https://docs.cartesia.ai/user-guides/voice-control"""

    speed: Optional[Union[float, str]] = None
    emotion: Optional[List[str]] = None


class CartesiaSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.CARTESIA.value):  # type: ignore
    api_key: Optional[str] = None
    model_id: str = DEFAULT_CARTESIA_MODEL_ID
    voice_id: str = DEFAULT_CARTESIA_VOICE_ID
    experimental_voice_controls: Optional[CartesiaVoiceControls] = None
