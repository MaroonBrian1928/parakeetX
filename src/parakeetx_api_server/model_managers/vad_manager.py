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
        self._device: str = "cpu"
        self._lock = threading.Lock()

    def _resolve_device(self) -> str:
        configured = self._settings.device.lower().strip()
        if configured == "cpu":
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        if configured == "cuda":
            logger.warning("VAD device set to 'cuda' but CUDA is unavailable; falling back to CPU")
        return "cpu"

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
            "device": self._device,
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

            device = self._resolve_device()
            # ONNX wrapper here doesn't expose CUDA execution provider; force JIT on GPU.
            use_onnx = self._settings.use_onnx and device == "cpu"

            try:
                self._model = load_silero_vad(
                    onnx=use_onnx,
                    opset_version=self._settings.onnx_opset_version,
                )
                self._backend = "onnx" if use_onnx else "jit"
            except Exception as exc:
                if not use_onnx or not self._settings.onnx_fallback_to_jit:
                    raise RuntimeError("Unable to load Silero VAD model") from exc

                logger.warning("Unable to load Silero ONNX VAD; falling back to JIT: %s", exc)
                self._model = load_silero_vad(onnx=False)
                self._backend = "jit"

            if device == "cuda" and self._backend == "jit":
                try:
                    import torch

                    self._model.to(torch.device("cuda"))
                except Exception as exc:
                    logger.warning("Unable to move Silero VAD to CUDA; using CPU: %s", exc)
                    device = "cpu"

            self._device = device
            self._get_speech_timestamps = get_speech_timestamps
            return self.status()

    def unload_model(self) -> dict[str, Any]:
        with self._lock:
            self._model = None
            self._get_speech_timestamps = None
            self._backend = None
            self._device = "cpu"
        return self.status()

    def _batched_probs(self, audio_tensor: Any) -> Any | None:
        """Run silero in a single batched forward pass over fixed 512-sample windows.

        Returns a 1-D float tensor of per-window speech probabilities, or None if the
        model doesn't expose a batched path (in which case the caller falls back to
        the per-window streaming loop).
        """
        try:
            import torch
        except Exception:
            return None

        model = self._model
        if model is None:
            return None
        audio_forward = getattr(model, "audio_forward", None)
        if audio_forward is None:
            inner = getattr(model, "_model", None)
            audio_forward = getattr(inner, "audio_forward", None) if inner is not None else None
        if audio_forward is None:
            logger.info(
                "Silero VAD: audio_forward not available on %s; using streaming loop",
                type(model).__name__,
            )
            return None

        try:
            with torch.no_grad():
                probs = audio_forward(audio_tensor.unsqueeze(0), TARGET_SAMPLE_RATE)
            logger.info(
                "Silero VAD: batched audio_forward path (device=%s windows=%d)",
                self._device,
                int(probs.shape[-1]),
            )
            return probs.squeeze(0).detach().to("cpu")
        except Exception as exc:
            logger.warning("Silero audio_forward failed; falling back to streaming loop: %s", exc)
            return None

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
            if self._device == "cuda":
                audio_tensor = audio_tensor.to(torch.device("cuda"))

            probs = self._batched_probs(audio_tensor)
            if probs is not None:
                timestamps = _segments_from_probs(
                    probs,
                    audio_length=int(audio_tensor.shape[0]),
                    sample_rate=TARGET_SAMPLE_RATE,
                    threshold=options.vad_onset,
                    neg_threshold=options.vad_offset,
                    min_speech_duration_ms=options.min_speech_duration_ms,
                    max_speech_duration_s=options.chunk_size,
                    min_silence_duration_ms=options.min_silence_duration_ms,
                    speech_pad_ms=options.speech_pad_ms,
                )
            else:
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


def _segments_from_probs(
    probs: Any,
    *,
    audio_length: int,
    sample_rate: int,
    threshold: float,
    neg_threshold: float,
    min_speech_duration_ms: int,
    max_speech_duration_s: float,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
) -> list[dict[str, float]]:
    """Port of silero-vad's threshold→segments state machine. Operates on per-window
    probabilities rather than calling the model itself, so the model can be invoked
    once in a batched forward pass beforehand.
    """
    window_size = 512  # silero uses 512 samples per window at 16 kHz
    min_speech_samples = sample_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sample_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sample_rate * max_speech_duration_s - window_size - 2 * speech_pad_samples
    )
    min_silence_samples = sample_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sample_rate * 98 / 1000

    probs_list = probs.tolist() if hasattr(probs, "tolist") else list(probs)
    speeches: list[dict[str, float]] = []
    current: dict[str, float] = {}
    triggered = False
    temp_end = 0
    prev_end = 0
    next_start = 0

    for i, prob in enumerate(probs_list):
        sample_pos = window_size * i
        if prob >= threshold and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = sample_pos

        if prob >= threshold and not triggered:
            triggered = True
            current["start"] = sample_pos
            continue

        if triggered and (sample_pos - current["start"]) > max_speech_samples:
            if prev_end:
                current["end"] = prev_end
                speeches.append(current)
                current = {}
                if next_start < prev_end:
                    triggered = False
                else:
                    current["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current["end"] = sample_pos
                speeches.append(current)
                current = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

        if prob < neg_threshold and triggered:
            if not temp_end:
                temp_end = sample_pos
            if (sample_pos - temp_end) > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if (sample_pos - temp_end) < min_silence_samples:
                continue
            current["end"] = temp_end
            if (current["end"] - current["start"]) > min_speech_samples:
                speeches.append(current)
            current = {}
            prev_end = next_start = temp_end = 0
            triggered = False
            continue

    if current and (audio_length - current["start"]) > min_speech_samples:
        current["end"] = audio_length
        speeches.append(current)

    for i, s in enumerate(speeches):
        if i == 0:
            s["start"] = max(0, s["start"] - speech_pad_samples)
        if i != len(speeches) - 1:
            silence_dur = speeches[i + 1]["start"] - s["end"]
            if silence_dur < 2 * speech_pad_samples:
                s["end"] += silence_dur // 2
                speeches[i + 1]["start"] = max(
                    0, speeches[i + 1]["start"] - silence_dur // 2
                )
            else:
                s["end"] = min(audio_length, s["end"] + speech_pad_samples)
                speeches[i + 1]["start"] = max(
                    0, speeches[i + 1]["start"] - speech_pad_samples
                )
        else:
            s["end"] = min(audio_length, s["end"] + speech_pad_samples)

    return [
        {"start": s["start"] / sample_rate, "end": s["end"] / sample_rate}
        for s in speeches
    ]


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
