from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..config import ParakeetSettings
from ..log_filters import install_noisy_dependency_log_filters, suppress_noisy_dependency_streams
from ..memory import release_memory_to_os
from .idle_eviction import IdleModelEvictor
from .model_worker import ModelWorkerClient

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ParakeetModelManager:
    def __init__(
        self,
        settings: ParakeetSettings,
        *,
        idle_evict_minutes: float | None = None,
        worker_client: ModelWorkerClient | None = None,
    ) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._worker_client = worker_client
        self._worker_loaded = False
        self._lock = threading.Lock()
        self._idle_evictor = IdleModelEvictor(
            model_label="parakeet",
            idle_minutes=idle_evict_minutes,
            is_loaded=self._is_loaded,
            unload=self.unload_model,
        )

    @property
    def configured_model_name(self) -> str:
        return self._settings.model_name

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._is_loaded(),
            "model_name": self._settings.model_name,
            "device": self._settings.device,
            "idle_evict_minutes": self._idle_evictor.idle_minutes,
        }

    def load_model(self) -> dict[str, Any]:
        if self._worker_client is not None:
            status = self._worker_client.load_parakeet()
            self._worker_loaded = True
            self._idle_evictor.note_loaded()
            status["idle_evict_minutes"] = self._idle_evictor.idle_minutes
            return status

        with self._lock:
            if self._model is not None:
                return self.status()

            install_noisy_dependency_log_filters()
            try:
                with suppress_noisy_dependency_streams():
                    from nemo.collections.asr.models import ASRModel
                    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
            except ImportError as exc:
                raise RuntimeError(
                    "NeMo ASR dependencies are not installed. Install with `uv sync --extra nemo`."
                ) from exc

            save_restore_connector = _build_save_restore_connector(
                SaveRestoreConnector,
                torch_load_mmap=self._settings.torch_load_mmap,
            )
            if self._settings.use_extracted_nemo_cache:
                with suppress_noisy_dependency_streams():
                    model_file = ASRModel.from_pretrained(
                        self._settings.model_name,
                        return_model_file=True,
                    )
                extracted_dir = _ensure_extracted_nemo_cache(
                    Path(model_file),
                    SaveRestoreConnector=SaveRestoreConnector,
                )
                save_restore_connector.model_extracted_dir = str(extracted_dir)

            with suppress_noisy_dependency_streams():
                self._model = ASRModel.from_pretrained(
                    self._settings.model_name,
                    map_location=self._settings.device,
                    save_restore_connector=save_restore_connector,
                )

            self._configure_cuda_runtime(self._model)
            self._configure_decoding(self._model)

        self._idle_evictor.note_loaded()
        return self.status()

    def _configure_cuda_runtime(self, model: Any) -> None:
        if not self._settings.device.startswith("cuda"):
            return

        try:
            import torch

            device = torch.device(self._settings.device)
            if hasattr(model, "to"):
                model.to(device)

            if self._settings.cuda_half_precision and hasattr(model, "half"):
                model.half()
                logger.info("Enabled CUDA half precision for ASR model on %s.", self._settings.device)

            if self._settings.cuda_torch_compile:
                encoder = getattr(model, "encoder", None)
                if encoder is not None:
                    try:
                        model.encoder = torch.compile(encoder, mode="reduce-overhead", fullgraph=False)
                        logger.info("Applied torch.compile to ASR encoder on %s.", self._settings.device)
                    except Exception as compile_exc:
                        logger.warning("torch.compile on ASR encoder failed; continuing uncompiled: %s", compile_exc)
        except Exception as exc:
            logger.warning("Unable to fully configure CUDA runtime for ASR model: %s", exc)

    def _configure_decoding(self, model: Any) -> None:
        if not self._settings.device.startswith("cuda"):
            return
        if not self._settings.cuda_force_greedy_decoding:
            return

        decoding_cfg = getattr(getattr(model, "cfg", None), "decoding", None)
        strategy = getattr(decoding_cfg, "strategy", None)
        if strategy != "greedy_batch":
            return

        # Maxwell-era GPUs can fail in NeMo's batched CUDA-graph decoder path (invalid PTX/invalid argument).
        # Use non-batched greedy decoding on CUDA to keep GPU execution while avoiding that path.
        try:
            from omegaconf import open_dict

            with open_dict(model.cfg.decoding):
                model.cfg.decoding.strategy = "greedy"

            model.change_decoding_strategy(model.cfg.decoding, verbose=False)
            logger.warning(
                "Adjusted RNNT decoding strategy from 'greedy_batch' to 'greedy' for CUDA compatibility."
            )
        except Exception as exc:
            logger.warning("Unable to adjust RNNT decoding strategy for CUDA compatibility: %s", exc)

    def unload_model(self) -> dict[str, Any]:
        if self._worker_client is not None:
            self._worker_loaded = False
            self._idle_evictor.cancel()
            status = self._worker_client.unload_parakeet()
            release_memory_to_os()
            status["idle_evict_minutes"] = self._idle_evictor.idle_minutes
            return status

        with self._lock:
            self._model = None
            self._idle_evictor.cancel()
            if self._settings.device.startswith("cuda"):
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
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

        if self._worker_client is not None:
            with self._idle_evictor.use():
                result = self._worker_client.transcribe_parakeet(
                    audio_path,
                    language=language,
                )
                self._worker_loaded = True
                return result

        try:
            with self._idle_evictor.use():
                model = self._model
                if model is None:
                    self.load_model()
                    model = self._model

                if model is None:
                    raise RuntimeError("Parakeet model failed to load")

                install_noisy_dependency_log_filters()
                chunk_plan = self._resolve_chunk_plan(audio_path)
                self._log_chunk_plan(audio_path, chunk_plan)
                chunk_seconds = chunk_plan["chunk_seconds"]
                if chunk_seconds is None:
                    raw = model.transcribe([str(audio_path)], timestamps=True)
                    try:
                        return self._normalize_raw_result(raw)
                    finally:
                        del raw

                return self._transcribe_chunked(
                    model=model,
                    audio_path=audio_path,
                    chunk_seconds=chunk_seconds,
                )
        finally:
            release_memory_to_os(clear_cuda=self._settings.device.startswith("cuda"))

    def transcribe_regions(
        self,
        audio_path: Path,
        regions: list[dict[str, Any]],
        *,
        language: str | None = None,
    ) -> dict[str, Any]:
        if language and language.lower() not in {"en", "english"}:
            raise ValueError("Only English transcription is supported")

        if self._worker_client is not None:
            with self._idle_evictor.use():
                result = self._worker_client.transcribe_parakeet_regions(
                    audio_path,
                    regions,
                    language=language,
                )
                self._worker_loaded = True
                return result

        if not regions:
            return {
                "text": "",
                "words": [],
                "segments": [],
                "language": "en",
                "model": self._settings.model_name,
            }

        info = sf.info(str(audio_path))
        sample_rate = int(info.samplerate)
        total_frames = int(info.frames)
        chunk_plan = self._resolve_chunk_plan(audio_path)
        asr_regions = _coalesce_regions_for_asr(
            regions,
            max_region_seconds=chunk_plan["chunk_seconds"],
        )

        try:
            with self._idle_evictor.use():
                model = self._model
                if model is None:
                    self.load_model()
                    model = self._model

                if model is None:
                    raise RuntimeError("Parakeet model failed to load")

                install_noisy_dependency_log_filters()
                whole_file_region = _single_region_covers_whole_file(
                    asr_regions,
                    total_frames=total_frames,
                    sample_rate=sample_rate,
                )
                if whole_file_region:
                    raw = model.transcribe([str(audio_path)], timestamps=True)
                    try:
                        chunks = [(0.0, self._normalize_raw_result(raw))]
                    finally:
                        del raw
                else:
                    chunk_paths: list[Path] = []
                    chunk_offsets: list[float] = []
                    with tempfile.TemporaryDirectory(
                        prefix="parakeetx-vad-chunks-",
                        dir=str(audio_path.parent),
                    ) as tmpdir:
                        chunk_dir = Path(tmpdir)
                        for index, region in enumerate(asr_regions):
                            start_seconds = max(0.0, _safe_float(region.get("start", 0.0)))
                            end_seconds = max(
                                start_seconds,
                                _safe_float(region.get("end", start_seconds)),
                            )
                            start_frame = min(total_frames, max(0, int(start_seconds * sample_rate)))
                            end_frame = min(
                                total_frames,
                                max(start_frame, int(end_seconds * sample_rate)),
                            )
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

                            chunk_path = chunk_dir / f"vad_chunk_{index:04d}.wav"
                            sf.write(
                                str(chunk_path),
                                np.asarray(audio_chunk, dtype=np.float32),
                                sample_rate,
                                format="WAV",
                                subtype="PCM_16",
                            )
                            chunk_paths.append(chunk_path)
                            chunk_offsets.append(float(start_frame) / float(sample_rate))

                        if not chunk_paths:
                            return {
                                "text": "",
                                "words": [],
                                "segments": [],
                                "language": "en",
                                "model": self._settings.model_name,
                            }

                        _log_vad_batch_plan(
                            chunk_paths,
                            chunk_offsets,
                            device=self._settings.device,
                        )
                        batch_size = max(1, int(self._settings.cuda_vad_batch_size))
                        chunks = []
                        for batch_start in range(0, len(chunk_paths), batch_size):
                            batch_paths = chunk_paths[batch_start : batch_start + batch_size]
                            batch_offsets = chunk_offsets[batch_start : batch_start + batch_size]
                            raw = model.transcribe(
                                [str(chunk_path) for chunk_path in batch_paths],
                                timestamps=True,
                            )
                            try:
                                chunks.extend(
                                    (offset, self._normalize_raw_result(item))
                                    for offset, item in zip(batch_offsets, _iter_transcript_items(raw))
                                )
                            finally:
                                del raw
                            release_memory_to_os(
                                clear_cuda=self._settings.device.startswith("cuda")
                            )

                return self._merge_chunk_payloads(chunks)
        finally:
            release_memory_to_os(clear_cuda=self._settings.device.startswith("cuda"))

    def _transcribe_regions_local(
        self,
        audio_path: Path,
        regions: list[dict[str, Any]],
        *,
        language: str | None = None,
    ) -> dict[str, Any]:
        return self.transcribe_regions(audio_path, regions, language=language)

    def _is_loaded(self) -> bool:
        if self._worker_client is not None:
            return self._worker_loaded
        return self._model is not None

    def _resolve_chunk_seconds(self, audio_path: Path) -> int | None:
        return self._resolve_chunk_plan(audio_path)["chunk_seconds"]

    def _resolve_chunk_plan(self, audio_path: Path) -> dict[str, Any]:
        if not self._settings.device.startswith("cuda"):
            return {
                "chunk_seconds": None,
                "reason": "non_cuda_device",
                "duration_seconds": _audio_duration_seconds(audio_path),
                "chunk_policy": "non_cuda",
                "gpu_name": None,
                "free_gib": None,
                "total_gib": None,
            }
        if not self._settings.cuda_adaptive_chunking:
            return {
                "chunk_seconds": None,
                "reason": "adaptive_chunking_disabled",
                "duration_seconds": _audio_duration_seconds(audio_path),
                "chunk_policy": "disabled",
                "gpu_name": None,
                "free_gib": None,
                "total_gib": None,
            }

        duration_seconds = _audio_duration_seconds(audio_path)

        if (
            self._settings.cuda_chunk_seconds_override is not None
            and self._settings.cuda_chunk_seconds_override > 0
        ):
            chunk_seconds = self._settings.cuda_chunk_seconds_override
            if duration_seconds > 0:
                chunk_seconds = min(chunk_seconds, int(max(1.0, duration_seconds)))
            return {
                "chunk_seconds": max(1, int(chunk_seconds)),
                "reason": "override",
                "duration_seconds": duration_seconds,
                "chunk_policy": "override",
                "gpu_name": None,
                "free_gib": None,
                "total_gib": None,
            }

        available_gib, total_gib, gpu_name = self._cuda_memory_snapshot()
        if available_gib is None:
            return {
                "chunk_seconds": None,
                "reason": "memory_probe_failed",
                "duration_seconds": duration_seconds,
                "chunk_policy": "unknown",
                "gpu_name": gpu_name,
                "free_gib": None,
                "total_gib": total_gib,
            }

        chunk_seconds = _chunk_seconds_for_available_gib(available_gib)
        chunk_seconds = max(self._settings.cuda_chunk_min_seconds, chunk_seconds)
        chunk_seconds = min(self._settings.cuda_chunk_max_seconds, chunk_seconds)

        if duration_seconds > 0:
            chunk_seconds = min(chunk_seconds, int(max(1.0, duration_seconds)))
            if duration_seconds <= float(chunk_seconds):
                return {
                    "chunk_seconds": None,
                    "reason": "audio_shorter_than_chunk",
                    "duration_seconds": duration_seconds,
                    "chunk_policy": "memory_only",
                    "gpu_name": gpu_name,
                    "free_gib": available_gib,
                    "total_gib": total_gib,
                }

        return {
            "chunk_seconds": max(1, int(chunk_seconds)),
            "reason": "adaptive",
            "duration_seconds": duration_seconds,
            "chunk_policy": "memory_only",
            "gpu_name": gpu_name,
            "free_gib": available_gib,
            "total_gib": total_gib,
        }

    def _log_chunk_plan(self, audio_path: Path, plan: dict[str, Any]) -> None:
        message = (
            "ASR request chunk plan: file=%s duration=%.2fs device=%s gpu=%s "
            "policy=%s free_gib=%s total_gib=%s chunk_seconds=%s reason=%s"
        ) % (
            audio_path.name,
            float(plan.get("duration_seconds") or 0.0),
            self._settings.device,
            plan.get("gpu_name") or "unknown",
            plan.get("chunk_policy") or "unknown",
            _fmt_gib(plan.get("free_gib")),
            _fmt_gib(plan.get("total_gib")),
            plan.get("chunk_seconds"),
            plan.get("reason") or "unknown",
        )
        logger.info(message)
        print(message, file=sys.stderr, flush=True)

    def _available_cuda_memory_gib(self) -> float | None:
        free_gib, _, _ = self._cuda_memory_snapshot()
        return free_gib

    def _cuda_memory_snapshot(self) -> tuple[float | None, float | None, str | None]:
        try:
            import torch

            device = torch.device(self._settings.device)
            if device.type != "cuda":
                return None, None, None
            free_bytes, _ = torch.cuda.mem_get_info(device)
            props = torch.cuda.get_device_properties(device)
            total_bytes = getattr(props, "total_memory", None)
            gpu_name = getattr(props, "name", None)
            total_gib = (
                float(total_bytes) / (1024.0**3) if isinstance(total_bytes, (int, float)) and total_bytes > 0 else None
            )
            return float(free_bytes) / (1024.0**3), total_gib, str(gpu_name) if gpu_name else None
        except Exception as exc:
            logger.warning("Unable to inspect available CUDA memory: %s", exc)
            return None, None, None

    def _transcribe_chunked(
        self,
        *,
        model: Any,
        audio_path: Path,
        chunk_seconds: int,
    ) -> dict[str, Any]:
        info = sf.info(str(audio_path))
        sample_rate = int(info.samplerate)
        total_frames = int(info.frames)
        chunk_frames = max(1, int(sample_rate * chunk_seconds))
        overlap_frames = max(0, int(sample_rate * self._settings.cuda_chunk_overlap_seconds))

        if total_frames <= chunk_frames:
            raw = model.transcribe([str(audio_path)], timestamps=True)
            return self._normalize_raw_result(raw)

        chunks: list[tuple[float, dict[str, Any]]] = []
        with tempfile.TemporaryDirectory(prefix="parakeetx-chunks-", dir=str(audio_path.parent)) as tmpdir:
            chunk_dir = Path(tmpdir)
            start_frame = 0
            chunk_index = 0

            while start_frame < total_frames:
                end_frame = min(total_frames, start_frame + chunk_frames)
                audio_chunk, _ = sf.read(
                    str(audio_path),
                    start=start_frame,
                    stop=end_frame,
                    dtype="float32",
                    always_2d=False,
                )
                if audio_chunk.size == 0:
                    break

                chunk_path = chunk_dir / f"chunk_{chunk_index:04d}.wav"
                sf.write(
                    str(chunk_path),
                    np.asarray(audio_chunk, dtype=np.float32),
                    sample_rate,
                    format="WAV",
                    subtype="PCM_16",
                )

                raw_chunk = model.transcribe([str(chunk_path)], timestamps=True)
                try:
                    chunk_payload = self._normalize_raw_result(raw_chunk)
                    chunk_offset_seconds = float(start_frame) / float(sample_rate)
                    chunks.append((chunk_offset_seconds, chunk_payload))
                finally:
                    del raw_chunk

                if end_frame >= total_frames:
                    break

                next_start = end_frame - overlap_frames
                if next_start <= start_frame:
                    next_start = end_frame
                start_frame = next_start
                chunk_index += 1

        return self._merge_chunk_payloads(chunks)

    def _merge_chunk_payloads(
        self,
        chunks: list[tuple[float, dict[str, Any]]],
    ) -> dict[str, Any]:
        merged_words: list[dict[str, Any]] = []
        merged_segments: list[dict[str, Any]] = []
        text_parts: list[str] = []
        segment_id = 0

        for offset_seconds, payload in chunks:
            text = str(payload.get("text", "")).strip()
            if text:
                text_parts.append(text)

            for word in payload.get("words", []):
                merged_words.append(
                    {
                        "word": str(word.get("word", "")),
                        "start": _safe_float(word.get("start", 0.0)) + offset_seconds,
                        "end": _safe_float(word.get("end", 0.0)) + offset_seconds,
                        "confidence": word.get("confidence"),
                    }
                )

            for segment in payload.get("segments", []):
                merged_segments.append(
                    {
                        "id": segment_id,
                        "start": _safe_float(segment.get("start", 0.0)) + offset_seconds,
                        "end": _safe_float(segment.get("end", 0.0)) + offset_seconds,
                        "text": str(segment.get("text") or ""),
                    }
                )
                segment_id += 1

        return {
            "text": " ".join(part for part in text_parts if part).strip(),
            "words": merged_words,
            "segments": merged_segments,
            "language": "en",
            "model": self._settings.model_name,
        }

    def _normalize_raw_result(self, raw: Any) -> dict[str, Any]:
        item = _first_transcript_item(raw)

        text = ""
        words: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []

        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = str(item.get("text", ""))
            words = [self._normalize_word(w) for w in item.get("words", [])]
            segments = [self._normalize_segment(s) for s in item.get("segments", [])]

            timestamps = _transcript_timestamps(item)
            if timestamps:
                if not words:
                    words = [self._normalize_word(w) for w in timestamps.get("word", [])]
                if not segments:
                    segments = [
                        self._normalize_segment(s)
                        for s in timestamps.get("segment", [])
                    ]
        elif item is not None:
            # NeMo commonly returns Hypothesis objects when timestamps=True.
            # Their text/timing lives on attributes rather than dict keys.
            text = str(_get_field(item, "text", default="") or "")
            timestamps = _transcript_timestamps(item)
            if timestamps:
                words = [self._normalize_word(w) for w in timestamps.get("word", [])]
                segments = [
                    self._normalize_segment(s)
                    for s in timestamps.get("segment", [])
                ]

            if not words:
                raw_words = _get_field(item, "words", default=[])
                if isinstance(raw_words, list):
                    words = [self._normalize_word(w) for w in raw_words]

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
            return {
                "word": str(_get_field(word, "word", "text", default=word) or ""),
                "start": _safe_float(_get_field(word, "start", default=0.0)),
                "end": _safe_float(_get_field(word, "end", default=0.0)),
                "confidence": _get_field(word, "confidence", default=None),
            }

        start = _safe_float(word.get("start", word.get("start_offset", 0.0)))
        return {
            "word": str(word.get("word") or word.get("text") or ""),
            "start": start,
            "end": _safe_float(word.get("end", word.get("end_offset", start))),
            "confidence": word.get("confidence"),
        }

    def _normalize_segment(self, segment: Any) -> dict[str, Any]:
        if not isinstance(segment, dict):
            return {
                "id": int(_safe_float(_get_field(segment, "id", default=0))),
                "start": _safe_float(_get_field(segment, "start", default=0.0)),
                "end": _safe_float(_get_field(segment, "end", default=0.0)),
                "text": str(
                    _get_field(segment, "text", "sentence", "segment", default=segment) or ""
                ),
            }

        start = _safe_float(segment.get("start", segment.get("start_offset", 0.0)))
        return {
            "id": int(segment.get("id", 0)),
            "start": start,
            "end": _safe_float(segment.get("end", segment.get("end_offset", start))),
            "text": str(
                segment.get("text")
                or segment.get("sentence")
                or segment.get("segment")
                or ""
            ),
        }


def _first_transcript_item(raw: Any) -> Any:
    item = raw
    while isinstance(item, (list, tuple)) and item:
        item = item[0]
    return item


def _iter_transcript_items(raw: Any) -> list[Any]:
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def _transcript_timestamps(item: Any) -> dict[str, Any]:
    for field in ("timestamps", "timestamp", "timestep"):
        timestamps = _get_field(item, field, default=None)
        if isinstance(timestamps, dict):
            return timestamps
    return {}


def _get_field(item: Any, *names: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        for name in names:
            if name in item:
                return item[name]
        return default

    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _chunk_seconds_for_available_gib(available_gib: float) -> int:
    if available_gib >= 6.0:
        return 600
    if available_gib >= 4.5:
        return 450
    if available_gib >= 3.0:
        return 300
    if available_gib >= 1.5:
        return 150
    return 90


def _audio_duration_seconds(audio_path: Path) -> float:
    try:
        return float(sf.info(str(audio_path)).duration)
    except Exception:
        return 0.0


def _fmt_gib(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "n/a"


def _log_vad_batch_plan(
    chunk_paths: list[Path],
    chunk_offsets: list[float],
    *,
    device: str,
) -> None:
    if not chunk_paths:
        return
    durations = [_audio_duration_seconds(path) for path in chunk_paths]
    total_speech_seconds = sum(durations)
    max_duration = max(durations) if durations else 0.0
    first_offset = chunk_offsets[0] if chunk_offsets else 0.0
    last_offset = chunk_offsets[-1] if chunk_offsets else 0.0
    message = (
        "ASR VAD batch plan: chunks=%s speech_duration=%.2fs max_chunk=%.2fs "
        "device=%s first_offset=%.2fs last_offset=%.2fs"
    ) % (
        len(chunk_paths),
        total_speech_seconds,
        max_duration,
        device,
        first_offset,
        last_offset,
    )
    logger.info(message)
    print(message, file=sys.stderr, flush=True)


def _single_region_covers_whole_file(
    regions: list[dict[str, Any]],
    *,
    total_frames: int,
    sample_rate: int,
) -> bool:
    if len(regions) != 1:
        return False

    region = regions[0]
    start_frame = max(0, int(_safe_float(region.get("start", 0.0)) * sample_rate))
    end_frame = int(_safe_float(region.get("end", 0.0)) * sample_rate)
    frame_tolerance = max(1, int(sample_rate * 0.01))
    return start_frame <= frame_tolerance and end_frame >= total_frames - frame_tolerance


def _coalesce_regions_for_asr(
    regions: list[dict[str, Any]],
    *,
    max_region_seconds: int | None,
) -> list[dict[str, Any]]:
    if not regions or max_region_seconds is None:
        return regions

    max_seconds = float(max_region_seconds)
    coalesced: list[dict[str, Any]] = []
    current_start: float | None = None
    current_end: float | None = None
    current_segments: list[tuple[float, float]] = []

    for region in regions:
        start = max(0.0, _safe_float(region.get("start", 0.0)))
        end = max(start, _safe_float(region.get("end", start)))
        if end <= start:
            continue

        child_segments = _region_child_segments(region, start=start, end=end)
        if current_start is None or current_end is None:
            current_start = start
            current_end = end
            current_segments = child_segments
            continue

        if end - current_start > max_seconds and current_end > current_start:
            coalesced.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "segments": current_segments,
                }
            )
            current_start = start
            current_segments = child_segments
        else:
            current_segments.extend(child_segments)

        current_end = end

    if current_start is not None and current_end is not None and current_end > current_start:
        coalesced.append(
            {
                "start": current_start,
                "end": current_end,
                "segments": current_segments,
            }
        )

    return coalesced


def _region_child_segments(
    region: dict[str, Any],
    *,
    start: float,
    end: float,
) -> list[tuple[float, float]]:
    raw_segments = region.get("segments")
    if not isinstance(raw_segments, list):
        return [(start, end)]

    child_segments: list[tuple[float, float]] = []
    for item in raw_segments:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        child_start = max(start, _safe_float(item[0], start))
        child_end = min(end, max(child_start, _safe_float(item[1], child_start)))
        if child_end > child_start:
            child_segments.append((child_start, child_end))
    return child_segments or [(start, end)]


def _ensure_extracted_nemo_cache(
    nemo_file: Path,
    *,
    SaveRestoreConnector: Any,
) -> Path:
    extracted_dir = nemo_file.with_name(f"{nemo_file.name}.extracted")
    if _is_extracted_nemo_dir_complete(extracted_dir):
        return extracted_dir

    tmpdir = extracted_dir.with_name(f"{extracted_dir.name}.tmp")
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True)
    try:
        SaveRestoreConnector._unpack_nemo_file(str(nemo_file), str(tmpdir))
        if not _is_extracted_nemo_dir_complete(tmpdir):
            raise RuntimeError(f"Extracted NeMo cache is incomplete: {tmpdir}")
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir)
        tmpdir.rename(extracted_dir)
    except Exception:
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    logger.info("Prepared extracted NeMo cache at %s.", extracted_dir)
    return extracted_dir


def _is_extracted_nemo_dir_complete(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "model_config.yaml").is_file()
        and (path / "model_weights.ckpt").is_file()
    )


def _build_save_restore_connector(
    SaveRestoreConnector: Any,
    *,
    torch_load_mmap: bool,
) -> Any:
    if not torch_load_mmap:
        return SaveRestoreConnector()

    class MMapSaveRestoreConnector(SaveRestoreConnector):
        @staticmethod
        def _load_state_dict_from_disk(model_weights, map_location="cpu"):
            import torch

            try:
                return torch.load(
                    model_weights,
                    map_location=map_location,
                    weights_only=True,
                    mmap=True,
                )
            except TypeError:
                return torch.load(
                    model_weights,
                    map_location=map_location,
                    weights_only=True,
                )

    return MMapSaveRestoreConnector()
