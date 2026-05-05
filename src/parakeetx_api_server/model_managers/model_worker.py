from __future__ import annotations

import logging
import multiprocessing as mp
import threading
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from ..config import DiarizationSettings, ParakeetSettings, Settings

logger = logging.getLogger(__name__)


class ModelWorkerClient:
    def __init__(self, settings: Settings) -> None:
        self._settings_payload = {
            "parakeet": settings.parakeet.model_dump(),
            "diarization": settings.diarization.model_dump(),
            "hf_token": settings.hf_token,
        }
        self._lock = threading.Lock()
        self._process: mp.Process | None = None
        self._connection: Connection | None = None

    @property
    def is_running(self) -> bool:
        process = self._process
        return process is not None and process.is_alive()

    def parakeet_status(self) -> dict[str, Any]:
        if not self.is_running:
            return _unloaded_status(self._settings_payload["parakeet"])
        return self._request("parakeet_status")

    def load_parakeet(self) -> dict[str, Any]:
        return self._request("load_parakeet")

    def unload_parakeet(self) -> dict[str, Any]:
        if not self.is_running:
            return _unloaded_status(self._settings_payload["parakeet"])
        status = self._request("unload_parakeet")
        self.stop_if_unloaded()
        return status

    def transcribe_parakeet(
        self,
        audio_path: Path,
        *,
        language: str | None,
    ) -> dict[str, Any]:
        return self._request(
            "transcribe_parakeet",
            {"audio_path": str(audio_path), "language": language},
        )

    def diarization_status(self) -> dict[str, Any]:
        if not self.is_running:
            payload = _unloaded_status(self._settings_payload["diarization"])
            payload["requires_hf_token"] = True
            return payload
        return self._request("diarization_status")

    def load_diarization(self) -> dict[str, Any]:
        return self._request("load_diarization")

    def unload_diarization(self) -> dict[str, Any]:
        if not self.is_running:
            payload = _unloaded_status(self._settings_payload["diarization"])
            payload["requires_hf_token"] = True
            return payload
        status = self._request("unload_diarization")
        self.stop_if_unloaded()
        return status

    def diarize(
        self,
        audio_path: Path,
        *,
        min_speakers: int | None,
        max_speakers: int | None,
        num_speakers: int | None,
    ) -> list[dict[str, Any]]:
        return self._request(
            "diarize",
            {
                "audio_path": str(audio_path),
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "num_speakers": num_speakers,
            },
        )

    def stop_if_unloaded(self) -> None:
        if not self.is_running:
            return
        try:
            status = self._request("all_status")
        except RuntimeError:
            return
        if status["parakeet"]["loaded"] or status["diarization"]["loaded"]:
            return
        self.shutdown()

    def shutdown(self) -> None:
        with self._lock:
            connection = self._connection
            process = self._process
            if connection is not None and process is not None and process.is_alive():
                try:
                    connection.send(("shutdown", None))
                    connection.recv()
                except Exception:
                    pass
            if connection is not None:
                connection.close()
            if process is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            self._connection = None
            self._process = None

    def _request(self, command: str, payload: dict[str, Any] | None = None) -> Any:
        with self._lock:
            self._ensure_running_locked()
            if self._connection is None:
                raise RuntimeError("Model worker failed to start")
            try:
                self._connection.send((command, payload or {}))
                response = self._connection.recv()
            except (EOFError, BrokenPipeError, OSError) as exc:
                self._reset_locked()
                raise RuntimeError("Model worker exited unexpectedly") from exc

        ok, value = response
        if ok:
            return value
        error_type = value.get("type", "RuntimeError")
        message = value.get("message", "Model worker request failed")
        raise RuntimeError(f"{error_type}: {message}")

    def _ensure_running_locked(self) -> None:
        process = self._process
        if process is not None and process.is_alive() and self._connection is not None:
            return
        self._reset_locked()

        parent_connection, child_connection = mp.get_context("spawn").Pipe()
        process = mp.get_context("spawn").Process(
            target=_model_worker_main,
            args=(child_connection, self._settings_payload),
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._connection = parent_connection
        self._process = process

    def _reset_locked(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._connection = None
        self._process = None


def _model_worker_main(
    connection: Connection,
    settings_payload: dict[str, Any],
) -> None:
    parakeet_manager = None
    diarization_manager = None

    try:
        while True:
            command, payload = connection.recv()
            try:
                if command == "shutdown":
                    connection.send((True, None))
                    return

                if parakeet_manager is None or diarization_manager is None:
                    from .diarization_manager import DiarizationModelManager
                    from .parakeet_manager import ParakeetModelManager

                    parakeet_manager = ParakeetModelManager(
                        ParakeetSettings(**settings_payload["parakeet"]),
                    )
                    diarization_manager = DiarizationModelManager(
                        DiarizationSettings(**settings_payload["diarization"]),
                        settings_payload.get("hf_token"),
                        idle_evict_minutes=None,
                    )

                if command == "parakeet_status":
                    result = parakeet_manager.status()
                elif command == "load_parakeet":
                    result = parakeet_manager.load_model()
                elif command == "unload_parakeet":
                    result = parakeet_manager.unload_model()
                elif command == "transcribe_parakeet":
                    result = parakeet_manager.transcribe(
                        Path(payload["audio_path"]),
                        language=payload.get("language"),
                    )
                elif command == "diarization_status":
                    result = diarization_manager.status()
                elif command == "load_diarization":
                    result = diarization_manager.load_model()
                elif command == "unload_diarization":
                    result = diarization_manager.unload_model()
                elif command == "diarize":
                    result = diarization_manager.diarize(
                        Path(payload["audio_path"]),
                        min_speakers=payload.get("min_speakers"),
                        max_speakers=payload.get("max_speakers"),
                        num_speakers=payload.get("num_speakers"),
                    )
                elif command == "all_status":
                    result = {
                        "parakeet": parakeet_manager.status(),
                        "diarization": diarization_manager.status(),
                    }
                else:
                    raise RuntimeError(f"Unknown model worker command: {command}")
                connection.send((True, result))
            except Exception as exc:
                logger.exception("Model worker command failed: %s", command)
                connection.send((False, {"type": type(exc).__name__, "message": str(exc)}))
    except EOFError:
        return


def _unloaded_status(settings_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "loaded": False,
        "model_name": settings_payload["model_name"],
        "device": settings_payload.get("device"),
        "idle_evict_minutes": None,
    }
