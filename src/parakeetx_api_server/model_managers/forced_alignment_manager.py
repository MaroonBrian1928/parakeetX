from __future__ import annotations

import logging
import threading
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..config import ForcedAlignmentSettings
from .device_capability import (
    MIN_BF16_CAPABILITY,
    MIN_FLASH_ATTENTION_CAPABILITY,
    MIN_FP16_CAPABILITY,
    cuda_compute_capability,
    meets_capability,
)

logger = logging.getLogger(__name__)


_LANGUAGE_NAMES = {
    "en": "English",
    "english": "English",
    "zh": "Chinese",
    "chinese": "Chinese",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
    "it": "Italian",
    "italian": "Italian",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "es": "Spanish",
    "spanish": "Spanish",
}


@dataclass(frozen=True)
class ForcedAlignmentOptions:
    enabled: bool


class ForcedAlignmentModelManager:
    def __init__(self, settings: ForcedAlignmentSettings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def settings(self) -> ForcedAlignmentSettings:
        return self._settings

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._model is not None,
            "enabled": self._settings.enabled,
            "method": self._settings.method,
            "model_name": self._settings.model_name,
            "device": self._settings.device,
            "dtype": self._settings.dtype,
            "attn_implementation": self._settings.attn_implementation,
            "max_chunk_seconds": self._settings.max_chunk_seconds,
            "preload_model": self._settings.preload_model,
        }

    def load_model(self) -> dict[str, Any]:
        with self._lock:
            if self._model is not None:
                return self.status()

            try:
                import torch
                from qwen_asr import Qwen3ForcedAligner
            except ImportError as exc:
                raise RuntimeError(
                    "qwen-asr is not installed. Install with `uv sync --extra forced-alignment`."
                ) from exc

            kwargs: dict[str, Any] = {
                "dtype": _torch_dtype(torch, self._resolve_dtype()),
                "device_map": self._settings.device,
            }
            attn_implementation = self._resolve_attn_implementation()
            if attn_implementation:
                kwargs["attn_implementation"] = attn_implementation

            self._model = Qwen3ForcedAligner.from_pretrained(
                self._settings.model_name,
                **kwargs,
            )
            return self.status()

    def _resolve_dtype(self) -> str:
        dtype = self._settings.dtype
        minimum = {"bfloat16": MIN_BF16_CAPABILITY, "float16": MIN_FP16_CAPABILITY}.get(dtype)
        if minimum is None or meets_capability(self._settings.device, minimum):
            return dtype
        logger.warning(
            "Forced alignment dtype %s requires compute capability %s; %s reports %s. Using float32.",
            dtype,
            minimum,
            self._settings.device,
            cuda_compute_capability(self._settings.device),
        )
        return "float32"

    def _resolve_attn_implementation(self) -> str | None:
        attn_implementation = self._settings.attn_implementation
        if attn_implementation in {"flash_attention_2", "flash_attention_3"} and not meets_capability(
            self._settings.device, MIN_FLASH_ATTENTION_CAPABILITY
        ):
            logger.warning(
                "Forced alignment attn_implementation %s requires compute capability %s; "
                "%s reports %s. Using the default implementation.",
                attn_implementation,
                MIN_FLASH_ATTENTION_CAPABILITY,
                self._settings.device,
                cuda_compute_capability(self._settings.device),
            )
            return None
        return attn_implementation

    def unload_model(self) -> dict[str, Any]:
        with self._lock:
            self._model = None
        return self.status()

    def align(
        self,
        audio_path: Path,
        *,
        text: str,
        language: str | None,
    ) -> list[dict[str, Any]]:
        if not text.strip():
            return []
        if self._model is None:
            self.load_model()
        if self._model is None:
            raise RuntimeError("Qwen3 forced aligner did not load")

        results = self._model.align(
            audio=str(audio_path),
            text=text,
            language=_language_name(language),
        )
        if not results:
            return []

        return [_alignment_item_to_word(item) for item in results[0]]

    def align_segments(
        self,
        audio_path: Path,
        *,
        segments: list[dict[str, Any]],
        language: str | None,
    ) -> list[dict[str, Any]]:
        chunks = _coalesce_text_segments(
            segments,
            max_chunk_seconds=self._settings.max_chunk_seconds,
        )
        if not chunks:
            return []

        info = sf.info(str(audio_path))
        sample_rate = int(info.samplerate)
        total_frames = int(info.frames)
        aligned_words: list[dict[str, Any]] = []

        if self._model is None:
            self.load_model()
        if self._model is None:
            raise RuntimeError("Qwen3 forced aligner did not load")

        language_name = _language_name(language)
        batch_size = max(1, int(self._settings.batch_size))

        with tempfile.TemporaryDirectory(
            prefix="parakeetx-qwen-align-",
            dir=str(audio_path.parent),
        ) as tmpdir:
            chunk_dir = Path(tmpdir)
            prepared: list[tuple[Path, str, float]] = []
            for index, chunk in enumerate(chunks):
                start_seconds = max(0.0, float(chunk["start"]))
                end_seconds = max(start_seconds, float(chunk["end"]))
                text = str(chunk["text"]).strip()
                if not text:
                    continue

                start_frame = min(total_frames, max(0, int(start_seconds * sample_rate)))
                end_frame = min(total_frames, max(start_frame, int(end_seconds * sample_rate)))
                if end_frame <= start_frame:
                    continue

                audio_chunk, _ = sf.read(
                    str(audio_path),
                    start=start_frame,
                    stop=end_frame,
                    dtype="float32",
                    always_2d=False,
                )
                if audio_chunk.size == 0:
                    continue

                chunk_path = chunk_dir / f"align_chunk_{index:04d}.wav"
                sf.write(
                    str(chunk_path),
                    np.asarray(audio_chunk, dtype=np.float32),
                    sample_rate,
                    format="WAV",
                    subtype="PCM_16",
                )
                prepared.append((chunk_path, text, float(start_frame) / float(sample_rate)))

            for batch_start in range(0, len(prepared), batch_size):
                batch = prepared[batch_start : batch_start + batch_size]
                audios = [str(p) for p, _, _ in batch]
                texts = [t for _, t, _ in batch]
                results = self._model.align(
                    audio=audios,
                    text=texts,
                    language=[language_name] * len(batch),
                )
                for (_, _, offset), result in zip(batch, results):
                    chunk_words = [_alignment_item_to_word(item) for item in result]
                    aligned_words.extend(_offset_words(chunk_words, offset=offset))

        return aligned_words


def _torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _language_name(language: str | None) -> str:
    if not language:
        return "English"
    return _LANGUAGE_NAMES.get(language.lower().strip(), language)


def _alignment_item_to_word(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        text = item.get("text", item.get("word", ""))
        start = item.get("start_time", item.get("start", 0.0))
        end = item.get("end_time", item.get("end", start))
        score = item.get("score", item.get("confidence"))
    else:
        text = getattr(item, "text", getattr(item, "word", ""))
        start = getattr(item, "start_time", getattr(item, "start", 0.0))
        end = getattr(item, "end_time", getattr(item, "end", start))
        score = getattr(item, "score", getattr(item, "confidence", None))

    word = {
        "word": str(text),
        "start": float(start),
        "end": float(end),
    }
    if score is not None:
        word["score"] = float(score)
    return word


def _coalesce_text_segments(
    segments: list[dict[str, Any]],
    *,
    max_chunk_seconds: float,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = max(start, float(segment.get("end", start)))

        if current is None:
            current = {"start": start, "end": end, "texts": [text]}
            continue

        proposed_end = max(float(current["end"]), end)
        if proposed_end - float(current["start"]) > max_chunk_seconds:
            chunks.append(_finalize_chunk(current))
            current = {"start": start, "end": end, "texts": [text]}
            continue

        current["end"] = proposed_end
        current["texts"].append(text)

    if current is not None:
        chunks.append(_finalize_chunk(current))

    return chunks


def _finalize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": float(chunk["start"]),
        "end": float(chunk["end"]),
        "text": " ".join(str(text).strip() for text in chunk["texts"] if str(text).strip()),
    }


def _offset_words(words: list[dict[str, Any]], *, offset: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for word in words:
        item = dict(word)
        item["start"] = float(item.get("start", 0.0)) + offset
        item["end"] = float(item.get("end", item["start"])) + offset
        output.append(item)
    return output
