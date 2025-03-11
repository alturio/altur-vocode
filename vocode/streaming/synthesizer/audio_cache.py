from typing import Optional
import time
from loguru import logger

from vocode.streaming.utils.redis import initialize_redis_bytes
from vocode.streaming.utils.singleton import Singleton


class AudioCache(Singleton):
    def __init__(self):
        self.redis = initialize_redis_bytes()
        self.disabled = False
        self.max_cache_size = 1024 * 1024 * 1024  # 1GB
        self.default_ttl = 3600 * 4  # 4 hours
        self.cache_info_key = "audio_cache:info"  # LRU info
        self.size_key = "audio_cache:total_size"  # total size

    @staticmethod
    async def safe_create():
        if AudioCache in Singleton._instances:
            return Singleton._instances[AudioCache]

        audio_cache = AudioCache()
        try:
            await audio_cache.redis.ping()
            if not await audio_cache.redis.exists(audio_cache.size_key):
                await audio_cache.redis.set(audio_cache.size_key, 0)
                await audio_cache.redis.delete(audio_cache.cache_info_key)
        except Exception:
            logger.warning("Redis ping failed on startup, disabling audio cache")
            audio_cache.disabled = True
        return audio_cache

    def configure(
        self, max_cache_size: Optional[int] = None, default_ttl: Optional[int] = None
    ):
        """Configure cache parameters"""
        if max_cache_size is not None:
            self.max_cache_size = max_cache_size
        if default_ttl is not None:
            self.default_ttl = default_ttl

    def get_audio_key(self, voice_identifier: str, text: str) -> str:
        return f"audio_cache:{voice_identifier}:{text}"

    async def get_audio(self, voice_identifier: str, text: str) -> Optional[bytes]:
        if self.disabled:
            return None

        audio_key = self.get_audio_key(voice_identifier, text)
        audio_data = await self.redis.get(audio_key)

        if audio_data:
            logger.info(f"Audio found in cache for {voice_identifier} {text}")
            await self.update_access_time(audio_key)
            return audio_data
        return None

    async def set_audio(
        self, voice_identifier: str, text: str, audio: bytes, ttl: Optional[int] = None
    ):
        if self.disabled:
            logger.warning("Audio cache is disabled")
            return

        logger.info(f"Setting audio for {voice_identifier} {text}")
        audio_key = self.get_audio_key(voice_identifier, text)

        existing_size = 0
        existing_size_str = await self.redis.hget(
            self.cache_info_key, f"{audio_key}:size"
        )
        if existing_size_str:
            existing_size = int(existing_size_str)
            await self.redis.decrby(self.size_key, existing_size)

        audio_size = len(audio)
        await self.ensure_cache_size(audio_size)

        await self.redis.set(audio_key, audio)

        actual_ttl = ttl if ttl is not None else self.default_ttl
        await self.redis.expire(audio_key, actual_ttl)

        await self.update_access_time(audio_key)
        await self.update_cache_size(audio_key, audio_size)

    async def update_access_time(self, audio_key: str):
        """Update the last access time and increment access count for an item"""
        current_time = time.time()
        await self.redis.hset(
            self.cache_info_key, f"{audio_key}:last_access", current_time
        )
        await self.redis.hincrby(self.cache_info_key, f"{audio_key}:popularity", 1)

    async def update_cache_size(self, audio_key: str, size: int):
        """Track the size of an item in the cache"""
        await self.redis.hset(self.cache_info_key, f"{audio_key}:size", size)
        await self.redis.incrby(self.size_key, size)

    async def ensure_cache_size(self, new_item_size: int):
        """Make room in the cache if needed using LRU eviction"""
        current_size = int(await self.redis.get(self.size_key) or 0)

        if current_size + new_item_size > self.max_cache_size:
            logger.warning(
                f"Cache size would exceed limit. Current: {current_size}, New item: {new_item_size}, Max: {self.max_cache_size}"
            )
        return

    async def clear_cache(self):
        """Clear the entire cache"""
        if self.disabled:
            return

        keys = await self.redis.keys("audio_cache:*")
        if keys:
            await self.redis.delete(*keys)

        await self.redis.set(self.size_key, 0)
        logger.info("Audio cache cleared")
