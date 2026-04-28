from __future__ import annotations

import asyncio
import tempfile
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
        suffix = Path(upload.filename or "upload.wav").suffix or ".wav"

        with tempfile.TemporaryDirectory(prefix="parakeetx-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            source_path = tmpdir_path / f"input{suffix}"
            normalized_path = tmpdir_path / "normalized.wav"

            contents = await upload.read()
            source_path.write_bytes(contents)

            await asyncio.to_thread(normalize_audio_to_wav, source_path, normalized_path)

            async with self._semaphore:
                asr_payload = await asyncio.to_thread(
                    self._parakeet_manager.transcribe,
                    normalized_path,
                    language=language,
                )

                diarization_segments: list[dict[str, Any]] = []
                if diarize:
                    diarization_segments = await asyncio.to_thread(
                        self._diarization_manager.diarize,
                        normalized_path,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        num_speakers=num_speakers,
                    )

                words, segments = assign_speakers(
                    list(asr_payload.get("words", [])),
                    list(asr_payload.get("segments", [])),
                    diarization_segments,
                )

                return {
                    "text": asr_payload.get("text", ""),
                    "language": asr_payload.get("language", "en"),
                    "model": asr_payload.get("model"),
                    "words": words,
                    "segments": segments,
                    "diarization": diarization_segments,
                }
