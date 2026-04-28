from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from parakeetx_api_server.config import ParakeetSettings
from parakeetx_api_server.model_managers.parakeet_manager import (
    ParakeetModelManager,
    _chunk_seconds_for_available_gib,
    _chunk_seconds_for_gpu_profile,
    _gpu_chunk_profile,
)


def _write_silent_wav(path: Path, *, duration_seconds: float, sample_rate: int = 16_000) -> None:
    frames = max(1, int(duration_seconds * sample_rate))
    samples = np.zeros(frames, dtype=np.float32)
    sf.write(str(path), samples, sample_rate, format="WAV", subtype="PCM_16")


def test_chunk_seconds_for_available_gib_thresholds() -> None:
    assert _chunk_seconds_for_available_gib(2.9) == 30
    assert _chunk_seconds_for_available_gib(3.0) == 60
    assert _chunk_seconds_for_available_gib(6.0) == 90
    assert _chunk_seconds_for_available_gib(10.0) == 150
    assert _chunk_seconds_for_available_gib(16.0) == 240
    assert _chunk_seconds_for_available_gib(24.0) == 360


def test_chunk_seconds_for_legacy_titan_profile() -> None:
    assert _chunk_seconds_for_gpu_profile(2.9, profile="legacy_titan") == 30
    assert _chunk_seconds_for_gpu_profile(3.0, profile="legacy_titan") == 90
    assert _chunk_seconds_for_gpu_profile(6.0, profile="legacy_titan") == 150
    assert _chunk_seconds_for_gpu_profile(8.0, profile="legacy_titan") == 180
    assert _chunk_seconds_for_gpu_profile(10.0, profile="legacy_titan") == 210


def test_gpu_chunk_profile_detection() -> None:
    assert _gpu_chunk_profile("NVIDIA GeForce GTX TITAN X", 12.0) == "legacy_titan"
    assert _gpu_chunk_profile("NVIDIA H100 PCIe", 80.0) == "modern_high_end"
    assert _gpu_chunk_profile("Unknown GPU", 48.0) == "modern_high_end"
    assert _gpu_chunk_profile("NVIDIA RTX 3070", 8.0) == "default"


def test_resolve_chunk_seconds_uses_available_cuda_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    _write_silent_wav(audio_path, duration_seconds=600.0)

    settings = ParakeetSettings(
        device="cuda",
        cuda_adaptive_chunking=True,
        cuda_chunk_min_seconds=30,
        cuda_chunk_max_seconds=360,
    )
    manager = ParakeetModelManager(settings)
    monkeypatch.setattr(manager, "_available_cuda_memory_gib", lambda: 5.5)
    monkeypatch.setattr(
        manager,
        "_cuda_memory_snapshot",
        lambda: (5.5, 12.0, "NVIDIA GeForce GTX TITAN X"),
    )

    assert manager._resolve_chunk_seconds(audio_path) == 90


def test_transcribe_chunked_merges_offsets(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    _write_silent_wav(audio_path, duration_seconds=5.0)

    settings = ParakeetSettings(
        device="cuda",
        cuda_adaptive_chunking=True,
        cuda_chunk_seconds_override=2,
        cuda_chunk_overlap_seconds=0.0,
    )
    manager = ParakeetModelManager(settings)

    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def transcribe(self, audio_paths, timestamps=True):
            _ = timestamps
            name = Path(audio_paths[0]).name
            self.calls.append(name)
            if name == "chunk_0000.wav":
                return {
                    "text": "hello",
                    "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
                    "segments": [{"id": 0, "start": 0.0, "end": 0.5, "text": "hello"}],
                }
            if name == "chunk_0001.wav":
                return {
                    "text": "world",
                    "words": [{"word": "world", "start": 0.0, "end": 0.5}],
                    "segments": [{"id": 0, "start": 0.0, "end": 0.5, "text": "world"}],
                }
            if name == "chunk_0002.wav":
                return {
                    "text": "again",
                    "words": [{"word": "again", "start": 0.0, "end": 0.5}],
                    "segments": [{"id": 0, "start": 0.0, "end": 0.5, "text": "again"}],
                }
            return {"text": "fallback", "words": [], "segments": []}

    fake_model = FakeModel()
    manager._model = fake_model

    payload = manager.transcribe(audio_path)

    assert payload["text"] == "hello world again"
    assert [segment["id"] for segment in payload["segments"]] == [0, 1, 2]
    assert payload["segments"][0]["start"] == pytest.approx(0.0)
    assert payload["segments"][1]["start"] == pytest.approx(2.0)
    assert payload["segments"][2]["start"] == pytest.approx(4.0)
    assert payload["words"][0]["start"] == pytest.approx(0.0)
    assert payload["words"][1]["start"] == pytest.approx(2.0)
    assert payload["words"][2]["start"] == pytest.approx(4.0)
    assert fake_model.calls == ["chunk_0000.wav", "chunk_0001.wav", "chunk_0002.wav"]


def test_normalize_raw_result_handles_nemo_hypothesis_object() -> None:
    settings = ParakeetSettings()
    manager = ParakeetModelManager(settings)

    hypothesis = types.SimpleNamespace(
        text="hello world",
        timestamp={
            "word": [
                {"word": "hello", "start": 0.1, "end": 0.4},
                {"word": "world", "start": 0.5, "end": 0.9},
            ],
            "segment": [
                {"segment": "hello world", "start": 0.1, "end": 0.9},
            ],
        },
    )

    payload = manager._normalize_raw_result([hypothesis])

    assert payload["text"] == "hello world"
    assert payload["words"] == [
        {"word": "hello", "start": 0.1, "end": 0.4, "confidence": None},
        {"word": "world", "start": 0.5, "end": 0.9, "confidence": None},
    ]
    assert payload["segments"] == [
        {"id": 0, "start": 0.1, "end": 0.9, "text": "hello world"},
    ]


def test_normalize_raw_result_unwraps_nested_nemo_result() -> None:
    settings = ParakeetSettings()
    manager = ParakeetModelManager(settings)

    hypothesis = types.SimpleNamespace(text="nested transcript", timestamp={})

    payload = manager._normalize_raw_result([[hypothesis]])

    assert payload["text"] == "nested transcript"
    assert payload["segments"] == [
        {"id": 0, "start": 0.0, "end": 0.0, "text": "nested transcript"},
    ]


def test_configure_cuda_runtime_moves_model_and_enables_half(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = ParakeetSettings(device="cuda", cuda_half_precision=True)
    manager = ParakeetModelManager(settings)

    class FakeModel:
        def __init__(self) -> None:
            self.moved_to = None
            self.half_called = False

        def to(self, device):
            self.moved_to = device
            return self

        def half(self):
            self.half_called = True
            return self

    fake_torch = types.SimpleNamespace(device=lambda value: f"device:{value}")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = FakeModel()
    manager._configure_cuda_runtime(model)

    assert model.moved_to == "device:cuda"
    assert model.half_called is True


def test_log_chunk_plan_emits_one_line(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    _write_silent_wav(audio_path, duration_seconds=30.0)

    settings = ParakeetSettings(device="cuda", cuda_adaptive_chunking=True)
    manager = ParakeetModelManager(settings)
    caplog.set_level("INFO")
    manager._log_chunk_plan(
        audio_path,
        {
            "chunk_seconds": 60,
            "reason": "adaptive",
            "duration_seconds": 30.0,
            "gpu_profile": "default",
            "gpu_name": "NVIDIA RTX 3070",
            "free_gib": 7.2,
            "total_gib": 8.0,
        },
    )

    assert "ASR request chunk plan" in caplog.text
    assert "chunk_seconds=60" in caplog.text
