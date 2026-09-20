import logging
import math
import time
from contextlib import aclosing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps


@dataclass
class ProviderScope:
    deadline: float
    providers: set[str] = field(default_factory=set)
    pending: int = 0
    uncertain: bool = False
    closed: bool = False
    joined: bool = False

    def __post_init__(self):
        for name in (
            "openai._base_client",
            "groq._base_client",
            "google.genai._api_client",
            "google.genai.models",
            "httpx",
            "httpcore.connection",
            "httpcore.http11",
            "httpcore.http2",
        ):
            logging.getLogger(name).addFilter(_ordinary_provider_log)

    def admit(self):
        if (
            self.closed
            or not math.isfinite(self.deadline)
            or time.monotonic() >= self.deadline
        ):
            raise RuntimeError("Provider admission closed")

    @property
    def confirmed(self):
        return (
            self.closed
            and self.joined
            and not self.uncertain
            and self.pending == 0
            and self.providers == {"llm", "stt", "tts"}
        )


@dataclass
class _ProviderRequest:
    completed: bool = False


current_provider_scope: ContextVar[ProviderScope | None] = ContextVar(
    "provider_scope", default=None
)


def _ordinary_provider_log(record):
    return current_provider_scope.get() is None


def register_provider(kind):
    scope = current_provider_scope.get()
    if scope is not None:
        scope.admit()
        scope.providers.add(kind)
    return scope


def complete_provider_request(request):
    if request is not None:
        request.completed = True


@contextmanager
def provider_request(scope):
    if scope is None:
        yield None
        return
    scope.admit()
    request = _ProviderRequest()
    scope.pending += 1
    try:
        yield request
    except Exception:
        raise RuntimeError("Provider completion unconfirmed") from None
    finally:
        scope.pending -= 1
        scope.uncertain |= not request.completed


def provider_operation(function):
    @wraps(function)
    async def tracked(*args, **kwargs):
        with provider_request(getattr(args[0], "_provider_scope", None)) as request:
            kwargs["_provider_request"] = request
            return await function(*args, **kwargs)

    return tracked


def provider_stream(function):
    @wraps(function)
    async def tracked(*args, **kwargs):
        with provider_request(getattr(args[0], "_provider_scope", None)) as request:
            kwargs["_provider_request"] = request
            async with aclosing(function(*args, **kwargs)) as stream:
                async for value in stream:
                    yield value

    return tracked
