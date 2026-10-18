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


def test_use_waits_for_an_eviction_already_in_progress() -> None:
    loaded = True
    unload_started = threading.Event()
    unload_finished = threading.Event()
    finished_before_use: list[bool] = []

    def unload() -> None:
        nonlocal loaded
        unload_started.set()
        time.sleep(0.1)
        loaded = False
        unload_finished.set()

    evictor = IdleModelEvictor(
        model_label="test",
        idle_minutes=0.0001,
        is_loaded=lambda: loaded,
        unload=unload,
    )

    evictor.note_loaded()
    assert unload_started.wait(timeout=1.0)
    with evictor.use():
        finished_before_use.append(unload_finished.is_set())

    assert finished_before_use == [True]
