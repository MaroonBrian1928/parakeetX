from __future__ import annotations

import sys
import types

from parakeetx_api_server.memory import release_memory_to_os


def test_release_memory_can_clear_cuda_cache(monkeypatch) -> None:
    calls: list[str] = []

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: calls.append("empty_cache"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    release_memory_to_os(clear_cuda=True)

    assert calls == ["empty_cache"]
