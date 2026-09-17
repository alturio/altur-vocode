import os

import httpx


def altur_auth_headers(url: str | httpx.URL) -> dict[str, str]:
    token = os.getenv("DJANGO_API_TOKEN", "")
    if not token:
        raise RuntimeError("DJANGO_API_TOKEN is not configured")
    origin = httpx.URL(
        f"http://{os.getenv('DJANGO_HOST', '127.0.0.1')}:{os.getenv('DJANGO_PORT', '8000')}"
    )
    target = httpx.URL(url)
    if target.userinfo or (target.scheme, target.host, target.port) != (
        origin.scheme,
        origin.host,
        origin.port,
    ):
        raise ValueError("Unexpected Django service origin")
    return {"Authorization": token}


def altur_service_auth(request: httpx.Request) -> httpx.Request:
    request.headers.update(altur_auth_headers(request.url))
    return request
