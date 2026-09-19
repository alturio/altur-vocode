import pytest
from fakeredis import FakeAsyncRedis, FakeServer
from pytest_mock import MockerFixture

from vocode.streaming.synthesizer.audio_cache import AudioCache
from vocode.streaming.utils.singleton import Singleton


@pytest.fixture(autouse=True)
def cleanup_singleton_audio_cache():
    Singleton._instances.pop(AudioCache, None)
    yield
    Singleton._instances.pop(AudioCache, None)


@pytest.mark.asyncio
async def test_set_and_get(mocker: MockerFixture):
    """Basic: Audio caching preserves content and isolates otherwise identical language keys."""
    fake_redis = FakeAsyncRedis()
    mocker.patch(
        "vocode.streaming.synthesizer.audio_cache.initialize_redis_bytes",
        return_value=fake_redis,
    )
    try:
        cache = await AudioCache.safe_create()
        assert await cache.get_audio("en", "voice_id", "text") is None
        await cache.set_audio("en", "voice_id", "text", b"chunk")
        assert await cache.get_audio("en", "voice_id", "text") == b"chunk"
        assert await cache.get_audio("es", "voice_id", "text") is None
    finally:
        await fake_redis.aclose()


@pytest.mark.asyncio
async def test_safe_create_set_and_get_disabled(mocker: MockerFixture):
    """Error Handling: An unavailable optional cache preserves synthesis without cache hits."""
    server = FakeServer()
    server.connected = False
    fake_redis = FakeAsyncRedis(server=server)
    mocker.patch(
        "vocode.streaming.synthesizer.audio_cache.initialize_redis_bytes",
        return_value=fake_redis,
    )
    try:
        cache = await AudioCache.safe_create()
        assert await cache.get_audio("en", "voice_id", "text") is None
        await cache.set_audio("en", "voice_id", "text", b"chunk")
        assert await cache.get_audio("en", "voice_id", "text") is None
    finally:
        await fake_redis.aclose()
