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
    _build_save_restore_connector,
    _ensure_extracted_nemo_cache,
    _is_extracted_nemo_dir_complete,
    _chunk_seconds_for_available_gib,
)


def _write_silent_wav(path: Path, *, duration_seconds: float, sample_rate: int = 16_000) -> None:
    frames = max(1, int(duration_seconds * sample_rate))
    samples = np.zeros(frames, dtype=np.float32)
    sf.write(str(path), samples, sample_rate, format="WAV", subtype="PCM_16")


def test_chunk_seconds_for_available_gib_thresholds() -> None:
    assert _chunk_seconds_for_available_gib(1.4) == 90
    assert _chunk_seconds_for_available_gib(1.5) == 150
    assert _chunk_seconds_for_available_gib(3.0) == 300
    assert _chunk_seconds_for_available_gib(4.5) == 450
    assert _chunk_seconds_for_available_gib(6.0) == 600
    assert _chunk_seconds_for_available_gib(24.0) == 600


def test_resolve_chunk_seconds_uses_available_cuda_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    _write_silent_wav(audio_path, duration_seconds=600.0)

    settings = ParakeetSettings(
        device="cuda",
        cuda_adaptive_chunking=True,
        cuda_chunk_min_seconds=30,
        cuda_chunk_max_seconds=1200,
    )
    manager = ParakeetModelManager(settings)
    monkeypatch.setattr(manager, "_available_cuda_memory_gib", lambda: 5.5)
    monkeypatch.setattr(
        manager,
        "_cuda_memory_snapshot",
        lambda: (5.5, 12.0, "NVIDIA GeForce GTX TITAN X"),
    )

    assert manager._resolve_chunk_seconds(audio_path) == 450


def test_resolve_chunk_seconds_default_cap_keeps_modern_cuda_chunks_conservative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    _write_silent_wav(audio_path, duration_seconds=1558.0)

    settings = ParakeetSettings(
        device="cuda",
        cuda_adaptive_chunking=True,
    )
    manager = ParakeetModelManager(settings)
    monkeypatch.setattr(
        manager,
        "_cuda_memory_snapshot",
        lambda: (9.92, 15.92, "NVIDIA GeForce RTX 5070 Ti"),
    )

    assert manager._resolve_chunk_seconds(audio_path) == 600


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
            "chunk_policy": "default",
            "gpu_name": "NVIDIA RTX 3070",
            "free_gib": 7.2,
            "total_gib": 8.0,
        },
    )

    assert "ASR request chunk plan" in caplog.text
    assert "chunk_seconds=60" in caplog.text


def test_process_isolated_parakeet_manager_delegates_to_worker(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    _write_silent_wav(audio_path, duration_seconds=1.0)

    class FakeWorker:
        def __init__(self) -> None:
            self.unloaded = False

        def parakeet_status(self):
            return {
                "loaded": False,
                "model_name": "nvidia/parakeet-tdt-0.6b-v2",
                "device": "cuda",
                "idle_evict_minutes": None,
            }

        def transcribe_parakeet(self, path, *, language):
            assert path == audio_path
            assert language is None
            return {"text": "hello", "words": [], "segments": [], "language": "en"}

        def unload_parakeet(self):
            self.unloaded = True
            return self.parakeet_status()

    worker = FakeWorker()
    manager = ParakeetModelManager(
        ParakeetSettings(device="cuda"),
        idle_evict_minutes=1,
        worker_client=worker,
    )

    assert manager.transcribe(audio_path)["text"] == "hello"
    assert manager.status()["idle_evict_minutes"] == 1
    manager.unload_model()
    assert worker.unloaded is True


def test_ensure_extracted_nemo_cache_reuses_complete_directory(tmp_path: Path) -> None:
    nemo_file = tmp_path / "model.nemo"
    nemo_file.write_bytes(b"fake")
    extracted = tmp_path / "model.nemo.extracted"
    extracted.mkdir()
    (extracted / "model_config.yaml").write_text("model: {}\n")
    (extracted / "model_weights.ckpt").write_bytes(b"weights")

    class FakeConnector:
        called = False

        @staticmethod
        def _unpack_nemo_file(path2file: str, out_folder: str) -> str:
            _ = path2file
            _ = out_folder
            FakeConnector.called = True
            return out_folder

    assert _ensure_extracted_nemo_cache(nemo_file, SaveRestoreConnector=FakeConnector) == extracted
    assert FakeConnector.called is False


def test_ensure_extracted_nemo_cache_extracts_when_missing(tmp_path: Path) -> None:
    nemo_file = tmp_path / "model.nemo"
    nemo_file.write_bytes(b"fake")

    class FakeConnector:
        @staticmethod
        def _unpack_nemo_file(path2file: str, out_folder: str) -> str:
            _ = path2file
            out = Path(out_folder)
            (out / "model_config.yaml").write_text("model: {}\n")
            (out / "model_weights.ckpt").write_bytes(b"weights")
            return out_folder

    extracted = _ensure_extracted_nemo_cache(nemo_file, SaveRestoreConnector=FakeConnector)

    assert extracted == tmp_path / "model.nemo.extracted"
    assert _is_extracted_nemo_dir_complete(extracted)


def test_build_save_restore_connector_can_disable_mmap() -> None:
    class FakeConnector:
        pass

    connector = _build_save_restore_connector(FakeConnector, torch_load_mmap=False)

    assert type(connector) is FakeConnector
