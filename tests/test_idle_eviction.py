from __future__ import annotations

import threading
import time

from parakeetx_api_server.model_managers.idle_eviction import IdleModelEvictor


def test_idle_evictor_unloads_after_configured_delay() -> None:
    loaded = True
    evicted = threading.Event()

    def unload() -> None:
        nonlocal loaded
        loaded = False
        evicted.set()

    evictor = IdleModelEvictor(
        model_label="test",
        idle_minutes=0.0001,
        is_loaded=lambda: loaded,
        unload=unload,
    )

    evictor.note_loaded()

    assert evicted.wait(timeout=1.0)
    assert loaded is False


def test_idle_evictor_waits_for_active_use_before_unloading() -> None:
    loaded = True
    evicted = threading.Event()

    def unload() -> None:
        nonlocal loaded
        loaded = False
        evicted.set()

    evictor = IdleModelEvictor(
        model_label="test",
        idle_minutes=0.0001,
        is_loaded=lambda: loaded,
        unload=unload,
    )

    evictor.note_loaded()
    with evictor.use():
        time.sleep(0.03)
        assert not evicted.is_set()

    assert evicted.wait(timeout=1.0)
    assert loaded is False
