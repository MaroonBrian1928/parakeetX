from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from parakeetx_api_server.config import VadSettings
from parakeetx_api_server.model_managers.vad_manager import (
    VadModelManager,
    VadOptions,
    merge_vad_segments,
)


def _write_wav(path: Path, *, duration_seconds: float, sample_rate: int = 16_000) -> None:
    frames = max(1, int(duration_seconds * sample_rate))
    samples = np.zeros(frames, dtype=np.float32)
    sf.write(str(path), samples, sample_rate, format="WAV", subtype="PCM_16")


def test_merge_vad_segments_matches_whisperx_chunk_boundary() -> None:
    merged = merge_vad_segments(
        [
            {"start": 0.0, "end": 8.0},
            {"start": 9.0, "end": 19.0},
            {"start": 20.0, "end": 31.0},
        ],
        chunk_size=30.0,
    )

    assert merged == [
        {"start": 0.0, "end": 19.0, "segments": [(0.0, 8.0), (9.0, 19.0)]},
        {"start": 20.0, "end": 31.0, "segments": [(20.0, 31.0)]},
    ]


def test_detect_uses_silero_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    _write_wav(audio_path, duration_seconds=2.0)
    calls: list[dict[str, object]] = []

    def fake_get_speech_timestamps(audio, model, **kwargs):
        calls.append({"audio": audio, "model": model, **kwargs})
        return [{"start": 0.1, "end": 0.8}, {"start": 1.0, "end": 1.6}]

    fake_torch = types.SimpleNamespace(from_numpy=lambda value: value)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    manager = VadModelManager(VadSettings())
    manager._model = object()
    manager._get_speech_timestamps = fake_get_speech_timestamps
    options = VadOptions.from_settings(
        VadSettings(),
        enabled=True,
        vad_onset=0.4,
        vad_offset=0.25,
        chunk_size=30.0,
        min_speech_duration_ms=100,
        min_silence_duration_ms=200,
        speech_pad_ms=50,
    )

    segments = manager.detect(audio_path, options)

    assert segments == [
        {
            "start": 0.1,
            "end": 1.6,
            "segments": [(0.1, 0.8), (1.0, 1.6)],
        }
    ]
    assert calls[0]["threshold"] == 0.4
    assert calls[0]["neg_threshold"] == 0.25
    assert calls[0]["sampling_rate"] == 16_000
    assert calls[0]["min_speech_duration_ms"] == 100
    assert calls[0]["min_silence_duration_ms"] == 200
    assert calls[0]["speech_pad_ms"] == 50
    assert calls[0]["max_speech_duration_s"] == 30.0


def test_load_model_uses_onnx_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_silero_vad(**kwargs):
        calls.append(kwargs)
        return object()

    fake_module = types.SimpleNamespace(
        get_speech_timestamps=lambda *args, **kwargs: [],
        load_silero_vad=fake_load_silero_vad,
    )
    monkeypatch.setitem(sys.modules, "silero_vad", fake_module)

    manager = VadModelManager(VadSettings(use_onnx=True, onnx_opset_version=15))
    status = manager.load_model()

    assert calls == [{"onnx": True, "opset_version": 15}]
    assert status["loaded"] is True
    assert status["backend"] == "onnx"
    assert status["use_onnx"] is True
    assert status["onnx_opset_version"] == 15


def test_load_model_falls_back_to_jit_when_onnx_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_silero_vad(**kwargs):
        calls.append(kwargs)
        if kwargs.get("onnx") is True:
            raise RuntimeError("onnx unavailable")
        return object()

    fake_module = types.SimpleNamespace(
        get_speech_timestamps=lambda *args, **kwargs: [],
        load_silero_vad=fake_load_silero_vad,
    )
    monkeypatch.setitem(sys.modules, "silero_vad", fake_module)

    manager = VadModelManager(VadSettings(use_onnx=True, onnx_fallback_to_jit=True))
    status = manager.load_model()

    assert calls == [{"onnx": True, "opset_version": 16}, {"onnx": False}]
    assert status["loaded"] is True
    assert status["backend"] == "jit"


def test_load_model_can_disable_onnx_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_silero_vad(**kwargs):
        raise RuntimeError("onnx unavailable")

    fake_module = types.SimpleNamespace(
        get_speech_timestamps=lambda *args, **kwargs: [],
        load_silero_vad=fake_load_silero_vad,
    )
    monkeypatch.setitem(sys.modules, "silero_vad", fake_module)

    manager = VadModelManager(VadSettings(use_onnx=True, onnx_fallback_to_jit=False))

    with pytest.raises(RuntimeError, match="Unable to load Silero VAD model"):
        manager.load_model()
