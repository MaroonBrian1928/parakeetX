from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from parakeetx_api_server.config import DiarizationSettings
from parakeetx_api_server.model_managers.diarization_manager import (
    DiarizationModelManager,
    _map_compact_diarization_to_original,
    _speech_intervals_from_vad_regions,
)


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
            raise AssertionError("status should not call the model worker")

        def _status(self):
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

        def diarize_regions(self, path, regions, *, min_speakers, max_speakers, num_speakers):
            assert path == audio_path
            assert regions == [{"start": 0.0, "end": 1.0}]
            assert min_speakers == 1
            assert max_speakers is None
            assert num_speakers is None
            return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

        def unload_diarization(self):
            self.unloaded = True
            return self._status()

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
    assert manager.diarize_regions(
        audio_path,
        [{"start": 0.0, "end": 1.0}],
        min_speakers=1,
    ) == [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    assert manager.status()["idle_evict_minutes"] == 1
    assert manager.status()["loaded"] is True
    manager.unload_model()
    assert worker.unloaded is True


def test_vad_region_helpers_use_child_speech_segments() -> None:
    regions = [
        {
            "start": 0.0,
            "end": 10.0,
            "segments": [(0.5, 1.0), (5.0, 6.0)],
        },
        {"start": 12.0, "end": 13.0},
    ]

    assert _speech_intervals_from_vad_regions(regions) == [
        (0.5, 1.0),
        (5.0, 6.0),
        (12.0, 13.0),
    ]


def test_compact_diarization_segments_map_back_to_original_timeline() -> None:
    mapped = _map_compact_diarization_to_original(
        [
            {"start": 0.2, "end": 0.5, "speaker": "SPEAKER_00"},
            {"start": 0.8, "end": 1.4, "speaker": "SPEAKER_01"},
        ],
        [
            {"compact_start": 0.0, "compact_end": 0.5, "original_start": 10.0},
            {"compact_start": 0.5, "compact_end": 1.5, "original_start": 20.0},
        ],
    )

    assert mapped == [
        {"start": 10.2, "end": 10.5, "speaker": "SPEAKER_00"},
        {"start": 20.3, "end": 20.9, "speaker": "SPEAKER_01"},
    ]


def test_diarize_regions_compacts_speech_once_and_remaps(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    samples = np.zeros(10 * 16000, dtype=np.float32)
    sf.write(str(audio_path), samples, 16000)

    manager = DiarizationModelManager(DiarizationSettings(), hf_token="token")

    def fake_diarize(path, *, min_speakers, max_speakers, num_speakers):
        _ = path
        assert min_speakers == 1
        assert max_speakers is None
        assert num_speakers is None
        return [{"start": 0.25, "end": 0.75, "speaker": "SPEAKER_00"}]

    monkeypatch.setattr(manager, "diarize", fake_diarize)

    segments = manager.diarize_regions(
        audio_path,
        [{"start": 3.0, "end": 4.0}],
        min_speakers=1,
    )

    assert segments == [{"start": 3.25, "end": 3.75, "speaker": "SPEAKER_00"}]
