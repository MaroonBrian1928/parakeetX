from __future__ import annotations

import threading
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..config import VadSettings

TARGET_SAMPLE_RATE = 16_000
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VadOptions:
    enabled: bool
    method: str
    vad_onset: float
    vad_offset: float
    chunk_size: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int

    @classmethod
    def from_settings(
        cls,
        settings: VadSettings,
        *,
        enabled: bool | None = None,
        method: str | None = None,
        vad_onset: float | None = None,
        vad_offset: float | None = None,
        chunk_size: float | None = None,
        min_speech_duration_ms: int | None = None,
        min_silence_duration_ms: int | None = None,
        speech_pad_ms: int | None = None,
    ) -> "VadOptions":
        resolved_method = (method or settings.method).lower().strip()
        if resolved_method != "silero":
            raise ValueError("Only silero VAD is supported")

        return cls(
            enabled=settings.enabled if enabled is None else enabled,
            method=resolved_method,
            vad_onset=settings.vad_onset if vad_onset is None else vad_onset,
            vad_offset=settings.vad_offset if vad_offset is None else vad_offset,
            chunk_size=settings.chunk_size if chunk_size is None else chunk_size,
            min_speech_duration_ms=(
                settings.min_speech_duration_ms
                if min_speech_duration_ms is None
                else min_speech_duration_ms
            ),
            min_silence_duration_ms=(
                settings.min_silence_duration_ms
                if min_silence_duration_ms is None
                else min_silence_duration_ms
            ),
            speech_pad_ms=settings.speech_pad_ms if speech_pad_ms is None else speech_pad_ms,
        )

    def validate(self) -> None:
        if not (0.0 < self.vad_onset < 1.0):
            raise ValueError("vad_onset must be between 0 and 1")
        if not (0.0 < self.vad_offset < 1.0):
            raise ValueError("vad_offset must be between 0 and 1")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.min_speech_duration_ms < 0:
            raise ValueError("min_speech_duration_ms must be greater than or equal to 0")
        if self.min_silence_duration_ms < 0:
            raise ValueError("min_silence_duration_ms must be greater than or equal to 0")
        if self.speech_pad_ms < 0:
            raise ValueError("speech_pad_ms must be greater than or equal to 0")


class VadModelManager:
    def __init__(self, settings: VadSettings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._get_speech_timestamps: Any | None = None
        self._backend: str | None = None
        self._lock = threading.Lock()

    @property
    def settings(self) -> VadSettings:
        return self._settings

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._model is not None,
            "method": self._settings.method,
            "enabled": self._settings.enabled,
            "preload_model": self._settings.preload_model,
            "backend": self._backend,
            "use_onnx": self._settings.use_onnx,
            "onnx_fallback_to_jit": self._settings.onnx_fallback_to_jit,
            "onnx_opset_version": self._settings.onnx_opset_version,
        }

    def load_model(self) -> dict[str, Any]:
        with self._lock:
            if self._model is not None and self._get_speech_timestamps is not None:
                return self.status()

            try:
                from silero_vad import get_speech_timestamps, load_silero_vad
            except ImportError as exc:
                raise RuntimeError(
                    "silero-vad is not installed. Install with `uv sync --extra vad`."
                ) from exc

            try:
                self._model = load_silero_vad(
                    onnx=self._settings.use_onnx,
                    opset_version=self._settings.onnx_opset_version,
                )
                self._backend = "onnx" if self._settings.use_onnx else "jit"
            except Exception as exc:
                if not self._settings.use_onnx or not self._settings.onnx_fallback_to_jit:
                    raise RuntimeError("Unable to load Silero VAD model") from exc

                logger.warning("Unable to load Silero ONNX VAD; falling back to JIT: %s", exc)
                self._model = load_silero_vad(onnx=False)
                self._backend = "jit"
            self._get_speech_timestamps = get_speech_timestamps
            return self.status()

    def unload_model(self) -> dict[str, Any]:
        with self._lock:
            self._model = None
            self._get_speech_timestamps = None
            self._backend = None
        return self.status()

    def detect(
        self,
        audio_path: Path,
        options: VadOptions,
    ) -> list[dict[str, Any]]:
        options.validate()
        if not options.enabled:
            return []
        if options.method != "silero":
            raise ValueError("Only silero VAD is supported")

        if self._model is None or self._get_speech_timestamps is None:
            self.load_model()

        waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        if int(sample_rate) != TARGET_SAMPLE_RATE:
            raise ValueError(f"VAD requires {TARGET_SAMPLE_RATE} Hz normalized audio")

        mono = np.asarray(waveform, dtype=np.float32).mean(axis=1)
        try:
            import torch

            audio_tensor = torch.from_numpy(mono)
            timestamps = self._get_speech_timestamps(
                audio_tensor,
                self._model,
                threshold=options.vad_onset,
                neg_threshold=options.vad_offset,
                sampling_rate=TARGET_SAMPLE_RATE,
                min_speech_duration_ms=options.min_speech_duration_ms,
                max_speech_duration_s=options.chunk_size,
                min_silence_duration_ms=options.min_silence_duration_ms,
                speech_pad_ms=options.speech_pad_ms,
                return_seconds=True,
            )
        finally:
            del mono
            del waveform

        speech_segments = [
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
            }
            for item in timestamps
            if float(item["end"]) > float(item["start"])
        ]
        return merge_vad_segments(speech_segments, chunk_size=options.chunk_size)


def merge_vad_segments(
    segments: list[dict[str, Any]],
    *,
    chunk_size: float,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    merged: list[dict[str, Any]] = []
    current_start = float(segments[0]["start"])
    current_end = float(segments[0]["end"])
    child_segments: list[tuple[float, float]] = [(current_start, current_end)]

    for segment in segments[1:]:
        start = float(segment["start"])
        end = float(segment["end"])
        if end - current_start > chunk_size and current_end - current_start > 0:
            merged.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "segments": child_segments,
                }
            )
            current_start = start
            child_segments = []

        current_end = end
        child_segments.append((start, end))

    merged.append(
        {
            "start": current_start,
            "end": current_end,
            "segments": child_segments,
        }
    )
    return merged
