from __future__ import annotations

import asyncio
import copy
import hashlib
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from ..model_managers.diarization_manager import DiarizationModelManager
from ..model_managers.forced_alignment_manager import ForcedAlignmentModelManager
from ..model_managers.parakeet_manager import ParakeetModelManager
from ..model_managers.vad_manager import VadModelManager, VadOptions
from ..memory import release_memory_to_os
from .audio import normalize_audio_to_wav
from .speaker_assignment import assign_speakers


@dataclass(frozen=True)
class _TranscriptionRequestKey:
    audio_sha256: str
    language: str | None
    diarize: bool
    min_speakers: int | None
    max_speakers: int | None
    num_speakers: int | None
    vad_options: VadOptions
    forced_alignment: bool


class TranscriptionService:
    _UPLOAD_CHUNK_BYTES = 1024 * 1024

    def __init__(
        self,
        *,
        parakeet_manager: ParakeetModelManager,
        diarization_manager: DiarizationModelManager,
        vad_manager: VadModelManager,
        forced_alignment_manager: ForcedAlignmentModelManager,
        max_concurrency: int,
        unload_asr_before_diarization: bool = False,
        unload_asr_before_forced_alignment: bool = False,
        empty_cuda_cache_after_stage: bool = False,
    ) -> None:
        self._parakeet_manager = parakeet_manager
        self._diarization_manager = diarization_manager
        self._vad_manager = vad_manager
        self._forced_alignment_manager = forced_alignment_manager
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._unload_asr_before_diarization = unload_asr_before_diarization
        self._unload_asr_before_forced_alignment = unload_asr_before_forced_alignment
        self._empty_cuda_cache_after_stage = empty_cuda_cache_after_stage
        self._inflight_lock = asyncio.Lock()
        self._inflight: dict[_TranscriptionRequestKey, asyncio.Task[dict[str, Any]]] = {}

    @staticmethod
    def _release_cuda_cache() -> None:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def configured_model_name(self) -> str:
        return self._parakeet_manager.configured_model_name

    async def _attach_or_start_inflight(
        self,
        *,
        key: _TranscriptionRequestKey,
        source_path: Path,
        normalized_path: Path,
        request_started: float,
        audio_hash_prefix: str,
    ) -> dict[str, Any]:
        task: asyncio.Task[dict[str, Any]]
        owner = False
        wait_started = time.perf_counter()

        async with self._inflight_lock:
            existing = self._inflight.get(key)
            if existing is not None and existing.done():
                self._inflight.pop(key, None)
                existing = None

            if existing is None:
                task = asyncio.create_task(
                    self._transcribe_source(
                        source_path=source_path,
                        normalized_path=normalized_path,
                        language=key.language,
                        diarize=key.diarize,
                        min_speakers=key.min_speakers,
                        max_speakers=key.max_speakers,
                        num_speakers=key.num_speakers,
                        vad_options=key.vad_options,
                        forced_alignment=key.forced_alignment,
                        request_started=request_started,
                    )
                )
                self._inflight[key] = task
                owner = True
            else:
                task = existing

        if not owner:
            result = await asyncio.shield(task)
            _emit_stage_timing(
                "dedupe_wait",
                wait_started,
                request_started=request_started,
                extra=f"attached=true sha256={audio_hash_prefix}",
            )
            return copy.deepcopy(result)

        try:
            result = await task
            return copy.deepcopy(result)
        finally:
            async with self._inflight_lock:
                if self._inflight.get(key) is task:
                    self._inflight.pop(key, None)

    async def transcribe_upload(
        self,
        *,
        upload: UploadFile,
        language: str | None,
        diarize: bool,
        min_speakers: int | None,
        max_speakers: int | None,
        num_speakers: int | None,
        vad_options: VadOptions,
        forced_alignment: bool,
    ) -> dict[str, Any]:
        request_started = time.perf_counter()
        suffix = Path(upload.filename or "upload.wav").suffix or ".wav"

        try:
            with tempfile.TemporaryDirectory(prefix="parakeetx-") as tmpdir:
                tmpdir_path = Path(tmpdir)
                source_path = tmpdir_path / f"input{suffix}"
                normalized_path = tmpdir_path / "normalized.wav"

                stage_started = time.perf_counter()
                hasher = hashlib.sha256()
                bytes_written = 0
                with source_path.open("wb") as source_file:
                    while chunk := await upload.read(self._UPLOAD_CHUNK_BYTES):
                        source_file.write(chunk)
                        hasher.update(chunk)
                        bytes_written += len(chunk)
                audio_sha256 = hasher.hexdigest()
                audio_hash_prefix = audio_sha256[:12]
                _emit_stage_timing(
                    "upload_read_write",
                    stage_started,
                    request_started=request_started,
                    extra=f"bytes={bytes_written} sha256={audio_hash_prefix}",
                )

                key = _TranscriptionRequestKey(
                    audio_sha256=audio_sha256,
                    language=language,
                    diarize=diarize,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    num_speakers=num_speakers,
                    vad_options=vad_options,
                    forced_alignment=forced_alignment,
                )
                return await self._attach_or_start_inflight(
                    key=key,
                    source_path=source_path,
                    normalized_path=normalized_path,
                    request_started=request_started,
                    audio_hash_prefix=audio_hash_prefix,
                )
        finally:
            release_memory_to_os(clear_cuda=True)

    async def _transcribe_source(
        self,
        *,
        source_path: Path,
        normalized_path: Path,
        language: str | None,
        diarize: bool,
        min_speakers: int | None,
        max_speakers: int | None,
        num_speakers: int | None,
        vad_options: VadOptions,
        forced_alignment: bool,
        request_started: float,
    ) -> dict[str, Any]:
        wait_started = time.perf_counter()
        async with self._semaphore:
            _emit_stage_timing(
                "slot_wait",
                wait_started,
                request_started=request_started,
                extra="acquired=true",
            )

            stage_started = time.perf_counter()
            await asyncio.to_thread(normalize_audio_to_wav, source_path, normalized_path)
            _emit_stage_timing(
                "audio_normalize",
                stage_started,
                request_started=request_started,
            )

            vad_segments: list[dict[str, Any]] = []
            if vad_options.enabled:
                stage_started = time.perf_counter()
                vad_segments = await asyncio.to_thread(
                    self._vad_manager.detect,
                    normalized_path,
                    vad_options,
                )
                _emit_stage_timing(
                    "vad",
                    stage_started,
                    request_started=request_started,
                    extra=f"segments={len(vad_segments)}",
                )

            stage_started = time.perf_counter()
            if vad_options.enabled:
                asr_payload = await asyncio.to_thread(
                    self._parakeet_manager.transcribe_regions,
                    normalized_path,
                    vad_segments,
                    language=language,
                )
            else:
                asr_payload = await asyncio.to_thread(
                    self._parakeet_manager.transcribe,
                    normalized_path,
                    language=language,
                )
            _emit_stage_timing(
                "asr",
                stage_started,
                request_started=request_started,
                extra=(
                    f"words={len(asr_payload.get('words', []))} "
                    f"segments={len(asr_payload.get('segments', []))}"
                ),
            )
            if self._empty_cuda_cache_after_stage:
                self._release_cuda_cache()

            if forced_alignment and self._forced_alignment_manager.settings.method == "qwen":
                if self._unload_asr_before_forced_alignment:
                    stage_started = time.perf_counter()
                    await asyncio.to_thread(self._parakeet_manager.unload_model)
                    _emit_stage_timing(
                        "asr_unload_before_forced_alignment",
                        stage_started,
                        request_started=request_started,
                    )

                stage_started = time.perf_counter()
                aligned_words = await asyncio.to_thread(
                    self._forced_alignment_manager.align_segments,
                    normalized_path,
                    segments=list(asr_payload.get("segments", [])),
                    language=language or asr_payload.get("language"),
                )
                if aligned_words:
                    asr_payload["words"] = aligned_words
                _emit_stage_timing(
                    "forced_alignment",
                    stage_started,
                    request_started=request_started,
                    extra=("method=qwen " f"words={len(asr_payload.get('words', []))}"),
                )
                if self._empty_cuda_cache_after_stage:
                    self._release_cuda_cache()
            elif forced_alignment:
                _emit_stage_timing(
                    "forced_alignment",
                    time.perf_counter(),
                    request_started=request_started,
                    extra=("method=parakeet " f"words={len(asr_payload.get('words', []))}"),
                )

            diarization_segments: list[dict[str, Any]] = []
            if diarize:
                if self._unload_asr_before_diarization:
                    stage_started = time.perf_counter()
                    await asyncio.to_thread(self._parakeet_manager.unload_model)
                    _emit_stage_timing(
                        "asr_unload_before_diarization",
                        stage_started,
                        request_started=request_started,
                    )

                stage_started = time.perf_counter()
                if vad_options.enabled:
                    diarization_segments = await asyncio.to_thread(
                        self._diarization_manager.diarize_regions,
                        normalized_path,
                        vad_segments,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        num_speakers=num_speakers,
                    )
                else:
                    diarization_segments = await asyncio.to_thread(
                        self._diarization_manager.diarize,
                        normalized_path,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        num_speakers=num_speakers,
                    )
                _emit_stage_timing(
                    "diarization",
                    stage_started,
                    request_started=request_started,
                    extra=(
                        f"segments={len(diarization_segments)} "
                        f"vad_compacted={str(vad_options.enabled).lower()}"
                    ),
                )
                if self._empty_cuda_cache_after_stage:
                    self._release_cuda_cache()

            stage_started = time.perf_counter()
            words, segments = assign_speakers(
                list(asr_payload.get("words", [])),
                list(asr_payload.get("segments", [])),
                diarization_segments,
            )
            _emit_stage_timing(
                "speaker_assignment",
                stage_started,
                request_started=request_started,
                extra=f"words={len(words)} segments={len(segments)}",
            )

            _emit_stage_timing("total", request_started, request_started=request_started)
            return {
                "text": asr_payload.get("text", ""),
                "language": asr_payload.get("language", "en"),
                "model": asr_payload.get("model"),
                "words": words,
                "segments": segments,
                "diarization": diarization_segments,
                "vad": vad_segments,
            }


def _emit_stage_timing(
    stage: str,
    started_at: float,
    *,
    request_started: float,
    extra: str | None = None,
) -> None:
    elapsed = time.perf_counter() - started_at
    total = time.perf_counter() - request_started
    suffix = f" {extra}" if extra else ""
    print(
        f"Transcription timing: stage={stage} elapsed={elapsed:.2f}s total={total:.2f}s{suffix}",
        file=sys.stderr,
        flush=True,
    )
