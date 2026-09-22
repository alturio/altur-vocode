import unittest

from vocode.streaming.telephony.config_manager.redis_config_manager import RedisConfigManager
from vocode.streaming.utils.redis import initialize_redis


class TestRedisConfigurationDatabase(unittest.IsolatedAsyncioTestCase):
    """Verify call configuration can select a database without moving other clients.

    Tests covered:
    - Call configuration accepts a nonzero database.
    - Default configuration and general-purpose clients retain database zero.
    """

    async def test_configuration_database_does_not_change_other_clients(self):
        """Critical: A configured call-state database leaves default Redis clients unchanged."""
        configured = RedisConfigManager(db=1).redis
        self.addAsyncCleanup(configured.aclose)
        default = RedisConfigManager().redis
        self.addAsyncCleanup(default.aclose)
        generic = initialize_redis()
        self.addAsyncCleanup(generic.aclose)

        self.assertEqual(configured.connection_pool.connection_kwargs["db"], 1)
        self.assertEqual(default.connection_pool.connection_kwargs["db"], 0)
        self.assertEqual(generic.connection_pool.connection_kwargs["db"], 0)
