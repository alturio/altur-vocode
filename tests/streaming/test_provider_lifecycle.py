import asyncio
import base64
import logging
import time
from contextlib import aclosing
from types import SimpleNamespace

import pytest
from vocode.streaming.action.external_actions_requester import ExternalActionsRequester
from vocode.streaming.utils.provider_lifecycle import (
    ProviderScope,
    complete_provider_request,
    current_provider_scope,
    provider_operation,
    provider_request,
    provider_stream,
    register_provider,
)


def scope():
    return ProviderScope(deadline=time.monotonic() + 30)


def test_only_completed_requests_plus_closed_joined_supported_components_confirm():
    """Critical: Local cleanup and absent requests cannot certify unsupported or incomplete provider work."""
    owner = scope()
    token = current_provider_scope.set(owner)
    try:
        for kind in ("llm", "stt", "tts"):
            assert register_provider(kind) is owner
        with provider_request(owner) as request:
            assert owner.pending == 1
            complete_provider_request(request)
            assert not owner.confirmed
        owner.closed = True
        assert not owner.confirmed
        owner.joined = True
        assert owner.confirmed
        with pytest.raises(RuntimeError, match="Provider admission closed"):
            with provider_request(owner):
                pytest.fail("closed admission ran provider code")
        assert owner.confirmed
    finally:
        current_provider_scope.reset(token)


def test_one_unconfirmed_request_is_permanent_despite_later_completion():
    """Critical: A successful retry cannot conceal earlier provider work with an unknown outcome."""
    owner = scope()
    owner.providers = {"llm", "stt", "tts"}
    with provider_request(owner):
        pass
    with provider_request(owner) as request:
        complete_provider_request(request)
    owner.closed = owner.joined = True
    assert owner.uncertain and owner.pending == 0 and not owner.confirmed


def test_managed_sdk_logs_cannot_emit_provider_bodies(caplog):
    """Critical: Managed SDK logs omit private bodies without changing ordinary-call logging."""
    token = current_provider_scope.set(scope())
    try:
        for name in (
            "openai._base_client",
            "groq._base_client",
            "google.genai.models",
            "httpx",
        ):
            logging.getLogger(name).warning("private provider body")
        assert "private provider body" not in caplog.text
    finally:
        current_provider_scope.reset(token)
    logging.getLogger("openai._base_client").warning("ordinary provider event")
    assert "ordinary provider event" in caplog.text


def test_provider_errors_are_sanitized_only_for_managed_requests():
    """Critical: Provider exceptions cannot expose raw bodies or certify interrupted work."""
    owner = scope()
    with pytest.raises(
        RuntimeError, match="^Provider completion unconfirmed$"
    ) as error:
        with provider_request(owner):
            raise ValueError("private provider body")
    assert error.value.__suppress_context__
    assert owner.uncertain and owner.pending == 0
    with pytest.raises(ValueError, match="private provider body"):
        with provider_request(None):
            raise ValueError("private provider body")


def test_expired_scope_denies_work_without_a_controller_poll():
    """Critical: Native monotonic expiry fences new provider requests while a watcher is delayed."""
    owner = scope()
    owner.deadline = time.monotonic() - 1
    with pytest.raises(RuntimeError, match="Provider admission closed"):
        with provider_request(owner):
            pytest.fail("expired scope admitted a request")
    assert owner.pending == 0


@pytest.mark.asyncio
async def test_stream_cleanup_uses_its_owner_even_in_a_different_task_context():
    """Race Condition: Generator finalization and ambient context changes cannot transfer request ownership."""
    first, other = scope(), scope()

    @provider_stream
    async def stream(self, _provider_request=None):
        yield "partial"
        complete_provider_request(_provider_request)

    generator = stream(SimpleNamespace(_provider_scope=first))
    assert await anext(generator) == "partial"
    assert first.pending == 1
    token = current_provider_scope.set(other)
    try:
        await asyncio.create_task(generator.aclose())
    finally:
        current_provider_scope.reset(token)
    assert first.pending == 0 and first.uncertain
    assert other.pending == 0 and not other.uncertain


@pytest.mark.asyncio
async def test_operations_keep_completion_and_cancellation_separate():
    """Verification: Explicit completion survives local cleanup; cancellation without a final response stays unknown."""
    first, other = scope(), scope()

    @provider_operation
    async def completed(self, _provider_request=None):
        complete_provider_request(_provider_request)

    @provider_operation
    async def cancelled(self, _provider_request=None):
        raise asyncio.CancelledError

    await completed(SimpleNamespace(_provider_scope=first))
    with pytest.raises(asyncio.CancelledError):
        await cancelled(SimpleNamespace(_provider_scope=other))
    assert first.pending == other.pending == 0
    assert not first.uncertain and other.uncertain


@pytest.mark.asyncio
async def test_complete_stream_and_ordinary_bypass():
    """Basic: A terminal response completes its owned request without imposing managed policy on ordinary agents."""
    owner = scope()

    @provider_stream
    async def stream(self, _provider_request=None):
        complete_provider_request(_provider_request)
        yield "complete"

    async with aclosing(stream(SimpleNamespace(_provider_scope=owner))) as generator:
        assert [value async for value in generator] == ["complete"]
    assert owner.pending == 0 and not owner.uncertain
    owner.closed = True
    assert [value async for value in stream(SimpleNamespace())] == ["complete"]


@pytest.mark.asyncio
async def test_external_mutations_remain_unknown_and_cannot_start_after_closure(
    httpx_mock,
):
    """Critical: Ordinary HTTP success is not an external mutation's terminal receipt and closed owners cannot retry it."""
    owner = scope()
    token = current_provider_scope.set(owner)
    try:
        requester = ExternalActionsRequester("https://action.example.test/")
    finally:
        current_provider_scope.reset(token)
    httpx_mock.add_response(url=requester.url, json={"result": {}})
    signature = base64.b64encode(b"test-only").decode()
    assert (await requester.send_request({}, signature)).success
    assert owner.uncertain
    owner.closed = True
    with pytest.raises(RuntimeError, match="Provider admission closed"):
        await requester.send_request({}, signature)
    assert len(httpx_mock.get_requests()) == 1
