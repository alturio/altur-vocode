import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from openai import NotFoundError

from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.agent.groq_agent import GroqAgent


class TestAgentClientLifecycle(unittest.IsolatedAsyncioTestCase):
    """Release per-agent clients without touching shared HTTP transports.

    Tests covered:
    - OpenAI and Groq termination closes their owned client
    - Independent cleanup proceeds after worker or vector-store failure
    - Model fallback closes the replaced client before using its replacement
    """

    async def test_owned_clients_close_after_worker_cleanup(self):
        """Critical: Each supported client-owning agent releases its own provider transport."""
        for agent_class, attribute in (
            (ChatGPTAgent, "openai_client"),
            (GroqAgent, "groq_client"),
        ):
            with self.subTest(agent=agent_class.__name__):
                agent = agent_class.__new__(agent_class)
                client = SimpleNamespace(close=AsyncMock())
                setattr(agent, attribute, client)
                with patch.object(
                    RespondAgent, "terminate", new=AsyncMock()
                ) as terminate:
                    await agent.terminate()
                terminate.assert_awaited_once()
                client.close.assert_awaited_once()

    async def test_worker_or_vector_cleanup_failure_cannot_skip_client_close(self):
        """Error Handling: Failure in independent cleanup cannot retain the provider client."""
        for worker_failure in (True, False):
            with self.subTest(worker_failure=worker_failure):
                agent = ChatGPTAgent.__new__(ChatGPTAgent)
                agent.openai_client = SimpleNamespace(close=AsyncMock())
                agent.vector_db = SimpleNamespace(
                    tear_down=AsyncMock(side_effect=RuntimeError("failed"))
                )
                terminate = AsyncMock(
                    side_effect=RuntimeError("failed") if worker_failure else None
                )
                with patch.object(RespondAgent, "terminate", new=terminate):
                    with self.assertRaises(RuntimeError):
                        await agent.terminate()
                agent.openai_client.close.assert_awaited_once()

    async def test_fallback_closes_the_replaced_client(self):
        """Verification: A successful provider fallback cannot leak the original HTTP client."""
        agent = ChatGPTAgent.__new__(ChatGPTAgent)
        response = httpx.Response(
            404, request=httpx.Request("POST", "https://example.test")
        )
        old = SimpleNamespace(
            close=AsyncMock(),
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(
                        side_effect=NotFoundError(
                            "missing", response=response, body=None
                        )
                    )
                )
            ),
        )
        new = SimpleNamespace(
            close=AsyncMock(),
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value="stream"))
            ),
        )
        agent.openai_client = old

        def replace(parameters):
            agent.openai_client = new

        agent.apply_model_fallback = MagicMock(side_effect=replace)
        self.assertEqual(
            await agent._create_openai_stream_with_fallback({"model": "test"}), "stream"
        )
        old.close.assert_awaited_once()
        new.close.assert_not_awaited()
        new.chat.completions.create.assert_awaited_once()
