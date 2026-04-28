from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from parakeetx_api_server.auth import require_api_key


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_api_key)])
    async def protected():
        return {"ok": True}

    return app


def test_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")

    from parakeetx_api_server.config import get_settings

    get_settings.cache_clear()

    client = TestClient(_app())
    r = client.get("/protected")
    assert r.status_code == 401

    r2 = client.get("/protected", headers={"Authorization": "Bearer secret"})
    assert r2.status_code == 200


def test_allows_without_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)

    from parakeetx_api_server.config import get_settings

    get_settings.cache_clear()

    client = TestClient(_app())
    r = client.get("/protected")
    assert r.status_code == 200
