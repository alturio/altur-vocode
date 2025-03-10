from typing import Optional, Dict
import time
from loguru import logger

from vocode.streaming.utils.redis import initialize_redis_bytes
from vocode.streaming.utils.singleton import Singleton


class AudioCache(Singleton):
    def __init__(self):
        self.redis = initialize_redis_bytes()
        self.disabled = False
        self.max_cache_size = 1024 * 1024 * 500  # 500MB
        self.default_ttl = 3600 * 1  # 1 hour
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
        """Update the last access time for an item (for LRU)"""
        current_time = time.time()
        await self.redis.hset(
            self.cache_info_key, f"{audio_key}:last_access", current_time
        )

    async def update_cache_size(self, audio_key: str, size: int):
        """Track the size of an item in the cache"""
        await self.redis.hset(self.cache_info_key, f"{audio_key}:size", size)
        await self.redis.incrby(self.size_key, size)

    async def ensure_cache_size(self, new_item_size: int):
        """Make room in the cache if needed using LRU eviction"""
        current_size = int(await self.redis.get(self.size_key) or 0)

        if current_size + new_item_size > self.max_cache_size:
            logger.info(
                f"Cache size would exceed limit. Current: {current_size}, New item: {new_item_size}, Max: {self.max_cache_size}"
            )
            await self.evict_lru_items(
                current_size + new_item_size - self.max_cache_size
            )

    async def evict_lru_items(self, bytes_to_free: int):
        """Evict least recently used items until we've freed enough space"""
        cache_info = await self.redis.hgetall(self.cache_info_key)
        if not cache_info:
            return

        items: Dict[str, Dict[str, float]] = {}
        for key, value in cache_info.items():
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            parts = key.rsplit(":", 1)
            if len(parts) == 2:
                item_key = parts[0]
                attribute = parts[1]

                if item_key not in items:
                    items[item_key] = {}

                items[item_key][attribute] = float(value)

        sorted_items = sorted(
            [(k, v) for k, v in items.items() if "last_access" in v and "size" in v],
            key=lambda x: x[1]["last_access"],
        )

        bytes_freed = 0
        for item_key, info in sorted_items:
            if bytes_freed >= bytes_to_free:
                break

            item_size = int(info["size"])

            await self.redis.delete(item_key)
            await self.redis.hdel(self.cache_info_key, f"{item_key}:last_access")
            await self.redis.hdel(self.cache_info_key, f"{item_key}:size")
            await self.redis.decrby(self.size_key, item_size)

            bytes_freed += item_size
            logger.info(f"Evicted {item_key} from cache (size: {item_size})")

        logger.info(f"Freed {bytes_freed} bytes from cache through LRU eviction")

    async def clear_cache(self):
        """Clear the entire cache"""
        if self.disabled:
            return

        keys = await self.redis.keys("audio_cache:*")
        if keys:
            await self.redis.delete(*keys)

        await self.redis.set(self.size_key, 0)
        logger.info("Audio cache cleared")
