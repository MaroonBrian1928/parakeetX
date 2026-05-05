from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..config import ParakeetSettings
from ..memory import release_memory_to_os

PARAKEET_GGUF_MODEL_NAME = "cstr/parakeet-tdt-0.6b-v3-GGUF"
PARAKEET_GGUF_MODEL_PATH = Path("/models/parakeet-tdt-0.6b-v3-q8_0.gguf")
CRISPASR_BINARY = "crispasr"
CRISPASR_TIMEOUT_SECONDS = 60 * 60


class ParakeetModelManager:
    def __init__(self, settings: ParakeetSettings) -> None:
        self._settings = settings

    @property
    def configured_model_name(self) -> str:
        return self._settings.model_name

    def status(self) -> dict[str, Any]:
        binary_path = _resolve_binary(CRISPASR_BINARY)
        model_exists = self._settings.model_path.is_file()
        return {
            "loaded": False,
            "available": binary_path is not None and model_exists,
            "backend": "crispasr",
            "model_name": self._settings.model_name,
            "model_path": str(self._settings.model_path),
            "binary": binary_path or CRISPASR_BINARY,
            "model_exists": model_exists,
        }

    def load_model(self) -> dict[str, Any]:
        self._validate_runtime()
        return self.status()

    def unload_model(self) -> dict[str, Any]:
        release_memory_to_os()
        return self.status()

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
    ) -> dict[str, Any]:
        if language and language.lower() not in {"en", "english"}:
            raise ValueError("Only English transcription is supported")

        self._validate_runtime()
        try:
            return _run_crispasr(
                binary=CRISPASR_BINARY,
                model_path=self._settings.model_path,
                audio_path=audio_path,
                model_name=self._settings.model_name,
            )
        finally:
            release_memory_to_os()

    def _validate_runtime(self) -> None:
        binary_path = _resolve_binary(CRISPASR_BINARY)
        if binary_path is None:
            raise RuntimeError(
                "CrispASR binary was not found on PATH. Install CrispASR and ensure `crispasr` is executable."
            )
        if not self._settings.model_path.is_file():
            raise RuntimeError(
                f"Parakeet GGUF model file not found at {self._settings.model_path}. "
                "Place parakeet-tdt-0.6b-v3-q8_0.gguf at /models before starting the API."
            )


def _run_crispasr(
    *,
    binary: str,
    model_path: Path,
    audio_path: Path,
    model_name: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="parakeetx-crispasr-") as tmpdir:
        output_base = Path(tmpdir) / "transcript"
        args = _build_crispasr_args(
            binary=binary,
            model_path=model_path,
            audio_path=audio_path,
            output_base=output_base,
        )
        env = os.environ.copy()
        env["CRISPASR_GGUF_MMAP"] = "1"

        completed = subprocess.run(
            args,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=CRISPASR_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "CrispASR transcription failed "
                f"(exit={completed.returncode}, stdout={_safe_process_text(completed.stdout)}, "
                f"stderr={_safe_process_text(completed.stderr)})"
            )

        json_path = output_base.with_suffix(".json")
        if not json_path.is_file():
            raise RuntimeError(f"CrispASR did not write expected JSON output: {json_path}")

        try:
            raw_payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"CrispASR wrote invalid JSON output: {json_path}") from exc

    return _normalize_crispasr_payload(raw_payload, model_name=model_name)


def _build_crispasr_args(
    *,
    binary: str,
    model_path: Path,
    audio_path: Path,
    output_base: Path,
) -> list[str]:
    args = [
        binary,
        "--backend",
        "parakeet",
        "-m",
        str(model_path),
        "-f",
        str(audio_path),
        "-ojf",
        "-of",
        str(output_base),
        "-np",
    ]
    gpu_backend = os.environ.get("CRISPASR_GPU_BACKEND", "").strip()
    if gpu_backend:
        args.extend(["--gpu-backend", gpu_backend])
    return args


def _normalize_crispasr_payload(payload: Any, *, model_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("CrispASR JSON output must be an object.")

    if isinstance(payload.get("transcription"), list):
        return _normalize_crispasr_transcription_payload(payload, model_name=model_name)

    text = str(
        payload.get("text")
        or payload.get("transcript")
        or payload.get("transcription")
        or ""
    )
    language = str(payload.get("language") or payload.get("lang") or "en")
    words = [_normalize_word(item) for item in _coerce_list(payload.get("words") or payload.get("tokens"))]
    segments = [
        _normalize_segment(item, index=index)
        for index, item in enumerate(
            _coerce_list(payload.get("segments") or payload.get("chunks") or payload.get("sentences"))
        )
    ]

    if not segments and text:
        end = words[-1]["end"] if words else 0.0
        segments = [{"id": 0, "start": 0.0, "end": end, "text": text}]

    return {
        "text": text,
        "language": language,
        "model": model_name,
        "words": words,
        "segments": segments,
    }


def _normalize_crispasr_transcription_payload(
    payload: dict[str, Any],
    *,
    model_name: str,
) -> dict[str, Any]:
    metadata = payload.get("crispasr")
    language = "en"
    if isinstance(metadata, dict):
        language = str(metadata.get("language") or metadata.get("lang") or "en")

    words: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for index, item in enumerate(_coerce_list(payload.get("transcription"))):
        if not isinstance(item, dict):
            continue

        segment = _normalize_transcription_segment(item, index=index)
        if segment["text"]:
            text_parts.append(segment["text"])
        segments.append(segment)

        for word in _coerce_list(item.get("words")):
            words.append(_normalize_word(word))

    text = str(payload.get("text") or " ".join(text_parts)).strip()
    return {
        "text": text,
        "language": language,
        "model": model_name,
        "words": words,
        "segments": segments,
    }


def _normalize_transcription_segment(segment: dict[str, Any], *, index: int) -> dict[str, Any]:
    offsets = segment.get("offsets")
    if isinstance(offsets, dict):
        start = _timestamp_milliseconds(offsets.get("from", 0.0))
        end = _timestamp_milliseconds(offsets.get("to", start * 1000.0))
    else:
        start = _timestamp_seconds(_first_present(segment, "start", "start_time", "t0"), default=0.0)
        end = _segment_end_seconds(segment, default=start)

    return {
        "id": index,
        "start": start,
        "end": end,
        "text": str(segment.get("text") or ""),
    }


def _normalize_word(word: Any) -> dict[str, Any]:
    if not isinstance(word, dict):
        return {"word": str(word), "start": 0.0, "end": 0.0, "confidence": None}

    start = _timestamp_seconds(_first_present(word, "start", "start_time", "t0"), default=0.0)
    if start == 0.0 and _first_present(word, "start_ms") is not None:
        start = _timestamp_milliseconds(word.get("start_ms"))
    return {
        "word": str(word.get("word") or word.get("text") or word.get("token") or ""),
        "start": start,
        "end": _word_end_seconds(word, default=start),
        "confidence": word.get("confidence", word.get("score")),
    }


def _normalize_segment(segment: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(segment, dict):
        return {"id": index, "start": 0.0, "end": 0.0, "text": str(segment)}

    start = _timestamp_seconds(_first_present(segment, "start", "start_time", "t0"), default=0.0)
    if start == 0.0 and _first_present(segment, "start_ms") is not None:
        start = _timestamp_milliseconds(segment.get("start_ms"))
    return {
        "id": int(_safe_float(segment.get("id", index), float(index))),
        "start": start,
        "end": _segment_end_seconds(segment, default=start),
        "text": str(segment.get("text") or segment.get("sentence") or segment.get("segment") or ""),
    }


def _word_end_seconds(word: dict[str, Any], *, default: float) -> float:
    value = _first_present(word, "end", "end_time", "t1")
    if value is not None:
        return _timestamp_seconds(value, default=default)
    if _first_present(word, "end_ms") is not None:
        return _timestamp_milliseconds(word.get("end_ms"))
    return default


def _segment_end_seconds(segment: dict[str, Any], *, default: float) -> float:
    value = _first_present(segment, "end", "end_time", "t1")
    if value is not None:
        return _timestamp_seconds(value, default=default)
    if _first_present(segment, "end_ms") is not None:
        return _timestamp_milliseconds(segment.get("end_ms"))
    return default


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _timestamp_seconds(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("ms"):
            return _safe_float(stripped[:-2]) / 1000.0
        if stripped.endswith("s"):
            return _safe_float(stripped[:-1])
        value = stripped

    numeric = _safe_float(value, default)
    if numeric > 10_000:
        return numeric / 1000.0
    return numeric


def _timestamp_milliseconds(value: Any) -> float:
    return _safe_float(value) / 1000.0


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _resolve_binary(binary: str) -> str | None:
    return shutil.which(binary)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_process_text(value: str | None, *, limit: int = 2000) -> str:
    text = (value or "").strip()
    if len(text) > limit:
        return f"{text[:limit]}...[truncated]"
    return text
