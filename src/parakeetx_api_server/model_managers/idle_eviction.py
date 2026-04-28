from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

logger = logging.getLogger(__name__)


class IdleModelEvictor:
    def __init__(
        self,
        *,
        model_label: str,
        idle_minutes: float | None,
        is_loaded: Callable[[], bool],
        unload: Callable[[], object],
    ) -> None:
        self._model_label = model_label
        self._idle_seconds = _idle_seconds(idle_minutes)
        self._is_loaded = is_loaded
        self._unload = unload
        self._lock = threading.RLock()
        self._active_uses = 0
        self._last_used = time.monotonic()
        self._timer: threading.Timer | None = None

    @property
    def idle_minutes(self) -> float | None:
        if self._idle_seconds is None:
            return None
        return self._idle_seconds / 60.0

    def note_loaded(self) -> None:
        with self._lock:
            self._last_used = time.monotonic()
            self._schedule_locked()

    def cancel(self) -> None:
        with self._lock:
            self._cancel_locked()

    @contextmanager
    def use(self) -> Iterator[None]:
        with self._lock:
            self._active_uses += 1
            self._cancel_locked()

        try:
            yield
        finally:
            with self._lock:
                self._active_uses = max(0, self._active_uses - 1)
                self._last_used = time.monotonic()
                self._schedule_locked()

    def _schedule_locked(self) -> None:
        if self._idle_seconds is None or self._active_uses > 0 or not self._is_loaded():
            return

        self._cancel_locked()
        delay = max(0.0, self._idle_seconds - (time.monotonic() - self._last_used))
        self._timer = threading.Timer(delay, self._evict_if_idle)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _evict_if_idle(self) -> None:
        with self._lock:
            if self._idle_seconds is None or self._active_uses > 0 or not self._is_loaded():
                self._timer = None
                return

            idle_for = time.monotonic() - self._last_used
            if idle_for < self._idle_seconds:
                self._schedule_locked()
                return

            self._timer = None

        logger.info(
            "Unloading %s model after %.2f minutes idle.",
            self._model_label,
            idle_for / 60.0,
        )
        self._unload()


def _idle_seconds(idle_minutes: float | None) -> float | None:
    if idle_minutes is None or idle_minutes <= 0:
        return None
    return float(idle_minutes) * 60.0
