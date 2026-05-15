from __future__ import annotations

import asyncio
import io
import shutil
import threading
import time
from types import SimpleNamespace

from fastapi import UploadFile

from parakeetx_api_server.model_managers.vad_manager import VadOptions
from parakeetx_api_server.services import transcription as transcription_module
from parakeetx_api_server.services.transcription import TranscriptionService


class _FakeParakeetManager:
    configured_model_name = "nvidia/parakeet-tdt-0.6b-v2"

    def __init__(self) -> None:
        self.call_count = 0
        self._lock = threading.Lock()

    def transcribe(self, path, *, language):
        _ = path
        _ = language
        with self._lock:
            self.call_count += 1
            call_number = self.call_count
        time.sleep(0.05)
        return {
            "text": f"call {call_number}",
            "language": "en",
            "model": self.configured_model_name,
            "words": [{"word": "hello", "start": 0.0, "end": 0.4}],
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hello"}],
        }

    def transcribe_regions(self, path, regions, *, language):
        _ = regions
        return self.transcribe(path, language=language)

    def unload_model(self):
        return {"loaded": False}


class _FakeDiarizationManager:
    def __init__(self) -> None:
        self.call_count = 0
        self._lock = threading.Lock()

    def diarize(self, path, *, min_speakers, max_speakers, num_speakers):
        _ = path
        _ = min_speakers
        _ = max_speakers
        _ = num_speakers
        with self._lock:
            self.call_count += 1
        return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

    def diarize_regions(self, path, regions, *, min_speakers, max_speakers, num_speakers):
        _ = regions
        return self.diarize(
            path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_speakers=num_speakers,
        )


class _FakeVadManager:
    def detect(self, path, options):
        _ = path
        _ = options
        return []


class _FakeForcedAlignmentManager:
    settings = SimpleNamespace(method="qwen")

    def align_segments(self, path, *, segments, language):
        _ = path
        _ = language
        return [
            {"word": segment["text"], "start": segment["start"], "end": segment["end"]}
            for segment in segments
        ]


def _make_service() -> tuple[TranscriptionService, _FakeParakeetManager, _FakeDiarizationManager]:
    parakeet = _FakeParakeetManager()
    diarization = _FakeDiarizationManager()
    return (
        TranscriptionService(
            parakeet_manager=parakeet,
            diarization_manager=diarization,
            vad_manager=_FakeVadManager(),
            forced_alignment_manager=_FakeForcedAlignmentManager(),
            max_concurrency=1,
        ),
        parakeet,
        diarization,
    )


def _upload(data: bytes) -> UploadFile:
    return UploadFile(filename="sample.wav", file=io.BytesIO(data))


def _vad_options() -> VadOptions:
    return VadOptions(
        enabled=False,
        method="silero",
        vad_onset=0.5,
        vad_offset=0.36,
        chunk_size=30.0,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=200,
    )


async def _transcribe(
    service: TranscriptionService,
    data: bytes,
    *,
    diarize: bool = False,
) -> dict:
    return await service.transcribe_upload(
        upload=_upload(data),
        language=None,
        diarize=diarize,
        min_speakers=None,
        max_speakers=None,
        num_speakers=None,
        vad_options=_vad_options(),
        forced_alignment=False,
    )


async def _gather_transcriptions(*coroutines):
    return await asyncio.gather(*coroutines)


def test_identical_inflight_uploads_share_the_same_work(monkeypatch):
    def fake_normalize(input_path, output_path):
        shutil.copyfile(input_path, output_path)
        return output_path

    monkeypatch.setattr(transcription_module, "normalize_audio_to_wav", fake_normalize)
    service, parakeet, diarization = _make_service()

    first, second = asyncio.run(
        _gather_transcriptions(
            _transcribe(service, b"same audio bytes"),
            _transcribe(service, b"same audio bytes"),
        )
    )

    assert first == second
    assert first is not second
    first["words"][0]["word"] = "mutated"
    assert second["words"][0]["word"] == "hello"
    assert parakeet.call_count == 1
    assert diarization.call_count == 0


def test_identical_audio_with_different_options_does_not_share_work(monkeypatch):
    def fake_normalize(input_path, output_path):
        shutil.copyfile(input_path, output_path)
        return output_path

    monkeypatch.setattr(transcription_module, "normalize_audio_to_wav", fake_normalize)
    service, parakeet, diarization = _make_service()

    plain, diarized = asyncio.run(
        _gather_transcriptions(
            _transcribe(service, b"same audio bytes", diarize=False),
            _transcribe(service, b"same audio bytes", diarize=True),
        )
    )

    assert {plain["text"], diarized["text"]} == {"call 1", "call 2"}
    assert plain["diarization"] == []
    assert diarized["diarization"] == [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    assert parakeet.call_count == 2
    assert diarization.call_count == 1
