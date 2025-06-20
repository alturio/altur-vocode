from typing import Dict, Optional
import time
from loguru import logger

from vocode.streaming.utils.redis import initialize_redis_bytes
from vocode.streaming.utils.singleton import Singleton


class AudioCache(Singleton):
    def __init__(self):
        self.redis = initialize_redis_bytes()
        self.disabled = False
        self.default_ttl = 3600 * 4  # 4 hours
        self.language_max_cache_size: Dict[str, int] = {
            "es": 1024 * 1024 * 1536,  # 1.5GB
            "en": 1024 * 1024 * 512,  # 512MB
            "pt": 1024 * 1024 * 512,  # 512MB
            "fr": 1024 * 1024 * 512,  # 512MB
            "default": 1024 * 1024 * 512,  # 512MB
        }

    @staticmethod
    async def safe_create():
        if AudioCache in Singleton._instances:
            return Singleton._instances[AudioCache]

        audio_cache = AudioCache()
        try:
            await audio_cache.redis.ping()
            for language in audio_cache.language_max_cache_size:
                if not await audio_cache.redis.exists(audio_cache.get_size_key(language)):
                    await audio_cache.redis.set(audio_cache.get_size_key(language), 0)
                    await audio_cache.redis.delete(audio_cache.get_cache_info_key(language))
        except Exception:
            logger.warning("Redis ping failed on startup, disabling audio cache")
            audio_cache.disabled = True
        return audio_cache

    def get_size_key(self, language: str) -> str:
        """Get the Redis key for tracking cache size for a specific language"""
        if language in self.language_max_cache_size:
            return f"audio_cache:size:{language}"
        return f"audio_cache:size:default"

    def get_cache_info_key(self, language: str) -> str:
        """Get the Redis key for tracking cache info (LRU data) for a specific language"""
        if language in self.language_max_cache_size:
            return f"audio_cache:info:{language}"
        return f"audio_cache:info:default"

    def get_audio_key(self, language: str, voice_identifier: str, text: str) -> str:
        """Generate a Redis key for an audio item based on language, voice identifier and text"""
        return f"audio_cache:{language}:{voice_identifier}:{text}"

    def get_max_cache_size(self, language: str) -> int:
        """Get the maximum cache size for a specific language"""
        if language in self.language_max_cache_size:
            return self.language_max_cache_size[language]
        return self.language_max_cache_size["default"]

    async def get_audio(self, language: str, voice_identifier: str, text: str) -> Optional[bytes]:
        if self.disabled:
            return None

        audio_key = self.get_audio_key(language, voice_identifier, text)
        audio_data = await self.redis.get(audio_key)

        if audio_data:
            logger.info(f"Audio found in cache for {language} {voice_identifier} {text}")
            await self.update_access_time(audio_key)
            return audio_data
        return None

    async def set_audio(
        self,
        language: str,
        voice_identifier: str,
        text: str,
        audio: bytes,
        ttl: Optional[int] = None,
    ):
        if self.disabled:
            logger.warning("Audio cache is disabled")
            return

        logger.info(f"Setting audio for {language} {voice_identifier} {text}")
        audio_key = self.get_audio_key(language, voice_identifier, text)

        existing_size = 0
        existing_size_str = await self.redis.hget(
            self.get_cache_info_key(language), f"{audio_key}:size"
        )
        if existing_size_str:
            existing_size = int(existing_size_str)
            await self.redis.decrby(self.get_size_key(language), existing_size)

        audio_size = len(audio)
        await self.ensure_cache_size(language, audio_size)

        await self.redis.set(audio_key, audio)

        actual_ttl = ttl if ttl is not None else self.default_ttl
        await self.redis.expire(audio_key, actual_ttl)

        await self.update_access_time(audio_key)
        await self.update_cache_size(audio_key, audio_size)

    async def update_access_time(self, audio_key: str):
        """Update the last access time and increment access count for an item"""
        try:
            language = audio_key.split(":")[1]
        except IndexError:
            logger.error(f"Invalid audio_key format: {audio_key}")
            return

        cache_info_key = self.get_cache_info_key(language)
        current_time = time.time()

        await self.redis.hset(cache_info_key, f"{audio_key}:last_access", current_time)
        await self.redis.hincrby(cache_info_key, f"{audio_key}:popularity", 1)

    async def update_cache_size(self, audio_key: str, size: int):
        """Track the size of an item in the cache"""
        try:
            language = audio_key.split(":")[1]
        except IndexError:
            logger.error(f"Invalid audio_key format: {audio_key}")
            return

        cache_info_key = self.get_cache_info_key(language)
        size_key = self.get_size_key(language)

        await self.redis.hset(cache_info_key, f"{audio_key}:size", size)
        await self.redis.incrby(size_key, size)

    async def ensure_cache_size(self, language: str, new_item_size: int):
        """Make room in the cache if needed using LRU eviction"""
        current_size = int(await self.redis.get(self.get_size_key(language)) or 0)

        if current_size + new_item_size > self.get_max_cache_size(language):
            logger.warning(
                f"Cache size would exceed limit for {language}. Current: {current_size}, New item: {new_item_size}, Max: {self.get_max_cache_size(language)}"
            )
        return

    async def clear_cache(self, language: str):
        """Clear the entire cache for a language"""
        if self.disabled:
            return

        keys = await self.redis.keys(f"audio_cache:{language}:*")
        if keys:
            await self.redis.delete(*keys)

        await self.redis.set(self.get_size_key(language), 0)
        await self.redis.delete(self.get_cache_info_key(language))

        logger.info(f"Audio cache cleared for {language}")
