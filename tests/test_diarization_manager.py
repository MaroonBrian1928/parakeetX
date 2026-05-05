from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from parakeetx_api_server.config import DiarizationSettings
from parakeetx_api_server.model_managers.diarization_manager import DiarizationModelManager


def test_diarization_uses_waveform_input(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    sf.write(str(audio_path), np.zeros(160, dtype=np.float32), 16000)
    calls: list[object] = []

    class FakeAnnotation:
        def itertracks(self, yield_label: bool):
            assert yield_label is True
            segment = type("Segment", (), {"start": 0.0, "end": 1.0})()
            yield segment, None, "SPEAKER_00"

    class FakePipeline:
        def __call__(self, audio_input, **kwargs):
            calls.append(audio_input)
            calls.append(kwargs)
            return FakeAnnotation()

    class FakeTensor:
        def unsqueeze(self, dim):
            assert dim == 0
            return self

    fake_torch = type("FakeTorch", (), {"from_numpy": lambda self, value: FakeTensor()})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    manager = DiarizationModelManager(DiarizationSettings(), hf_token="token")
    manager._pipeline = FakePipeline()

    result = manager.diarize(audio_path, min_speakers=1)

    assert isinstance(calls[0], dict)
    assert calls[1] == {"min_speakers": 1}
    assert result == [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]


def test_process_isolated_diarization_manager_delegates_to_worker(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    class FakeWorker:
        def __init__(self) -> None:
            self.unloaded = False

        def diarization_status(self):
            return {
                "loaded": False,
                "model_name": "pyannote/speaker-diarization-community-1",
                "device": "cuda",
                "idle_evict_minutes": None,
                "requires_hf_token": True,
            }

        def diarize(self, path, *, min_speakers, max_speakers, num_speakers):
            assert path == audio_path
            assert min_speakers == 1
            assert max_speakers is None
            assert num_speakers is None
            return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

        def unload_diarization(self):
            self.unloaded = True
            return self.diarization_status()

    worker = FakeWorker()
    manager = DiarizationModelManager(
        DiarizationSettings(device="cuda"),
        hf_token="token",
        idle_evict_minutes=1,
        worker_client=worker,
    )

    assert manager.diarize(audio_path, min_speakers=1) == [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}
    ]
    assert manager.status()["idle_evict_minutes"] == 1
    manager.unload_model()
    assert worker.unloaded is True
