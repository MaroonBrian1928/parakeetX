from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from ..model_managers.diarization_manager import DiarizationModelManager
from ..model_managers.parakeet_manager import ParakeetModelManager
from .audio import normalize_audio_to_wav
from .speaker_assignment import assign_speakers


class TranscriptionService:
    def __init__(
        self,
        *,
        parakeet_manager: ParakeetModelManager,
        diarization_manager: DiarizationModelManager,
        max_concurrency: int,
    ) -> None:
        self._parakeet_manager = parakeet_manager
        self._diarization_manager = diarization_manager
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))

    @property
    def configured_model_name(self) -> str:
        return self._parakeet_manager.configured_model_name

    async def transcribe_upload(
        self,
        *,
        upload: UploadFile,
        language: str | None,
        diarize: bool,
        min_speakers: int | None,
        max_speakers: int | None,
        num_speakers: int | None,
    ) -> dict[str, Any]:
        request_started = time.perf_counter()
        suffix = Path(upload.filename or "upload.wav").suffix or ".wav"

        with tempfile.TemporaryDirectory(prefix="parakeetx-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            source_path = tmpdir_path / f"input{suffix}"
            normalized_path = tmpdir_path / "normalized.wav"

            stage_started = time.perf_counter()
            contents = await upload.read()
            source_path.write_bytes(contents)
            _emit_stage_timing(
                "upload_read_write",
                stage_started,
                request_started=request_started,
                extra=f"bytes={len(contents)}",
            )

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

                stage_started = time.perf_counter()
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

                diarization_segments: list[dict[str, Any]] = []
                if diarize:
                    stage_started = time.perf_counter()
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
                        extra=f"segments={len(diarization_segments)}",
                    )

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
