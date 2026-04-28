from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..config import ParakeetSettings


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ParakeetModelManager:
    def __init__(self, settings: ParakeetSettings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def configured_model_name(self) -> str:
        return self._settings.model_name

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._model is not None,
            "model_name": self._settings.model_name,
            "device": self._settings.device,
        }

    def load_model(self) -> dict[str, Any]:
        with self._lock:
            if self._model is not None:
                return self.status()

            try:
                from nemo.collections.asr.models import ASRModel
            except ImportError as exc:
                raise RuntimeError(
                    "NeMo ASR dependencies are not installed. Install with `uv sync --extra nemo`."
                ) from exc

            self._model = ASRModel.from_pretrained(
                self._settings.model_name,
                map_location=self._settings.device,
            )

        return self.status()

    def unload_model(self) -> dict[str, Any]:
        with self._lock:
            self._model = None
            if self._settings.device.startswith("cuda"):
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
        return self.status()

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
    ) -> dict[str, Any]:
        if language and language.lower() not in {"en", "english"}:
            raise ValueError("Only English transcription is supported")

        model = self._model
        if model is None:
            self.load_model()
            model = self._model

        if model is None:
            raise RuntimeError("Parakeet model failed to load")

        raw = model.transcribe([str(audio_path)], timestamps=True)
        return self._normalize_raw_result(raw)

    def _normalize_raw_result(self, raw: Any) -> dict[str, Any]:
        item: Any = raw
        if isinstance(raw, list) and raw:
            item = raw[0]

        text = ""
        words: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []

        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = str(item.get("text", ""))
            words = [self._normalize_word(w) for w in item.get("words", [])]
            segments = [self._normalize_segment(s) for s in item.get("segments", [])]

            if not words and "timestamps" in item and isinstance(item["timestamps"], dict):
                words = [
                    self._normalize_word(w) for w in item["timestamps"].get("word", [])
                ]
            if not segments and "timestamps" in item and isinstance(item["timestamps"], dict):
                segments = [
                    self._normalize_segment(s)
                    for s in item["timestamps"].get("segment", [])
                ]

        if not segments and text:
            end = words[-1]["end"] if words else 0.0
            segments = [{"id": 0, "start": 0.0, "end": end, "text": text}]

        return {
            "text": text,
            "words": words,
            "segments": segments,
            "language": "en",
            "model": self._settings.model_name,
        }

    def _normalize_word(self, word: Any) -> dict[str, Any]:
        if not isinstance(word, dict):
            return {"word": str(word), "start": 0.0, "end": 0.0}

        return {
            "word": str(word.get("word") or word.get("text") or ""),
            "start": _safe_float(word.get("start", 0.0)),
            "end": _safe_float(word.get("end", 0.0)),
            "confidence": word.get("confidence"),
        }

    def _normalize_segment(self, segment: Any) -> dict[str, Any]:
        if not isinstance(segment, dict):
            return {"id": 0, "start": 0.0, "end": 0.0, "text": str(segment)}

        return {
            "id": int(segment.get("id", 0)),
            "start": _safe_float(segment.get("start", 0.0)),
            "end": _safe_float(segment.get("end", 0.0)),
            "text": str(segment.get("text") or segment.get("sentence") or ""),
        }
