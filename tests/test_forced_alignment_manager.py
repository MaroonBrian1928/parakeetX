from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from parakeetx_api_server.config import ForcedAlignmentSettings
from parakeetx_api_server.model_managers.forced_alignment_manager import (
    ForcedAlignmentModelManager,
    _alignment_item_to_word,
    _coalesce_text_segments,
    _offset_words,
    _language_name,
)


def test_language_name_normalizes_qwen_supported_languages() -> None:
    assert _language_name(None) == "English"
    assert _language_name("en") == "English"
    assert _language_name("Spanish") == "Spanish"


def test_alignment_item_to_word_accepts_qwen_objects_and_dicts() -> None:
    item = SimpleNamespace(text="hello", start_time=0.1, end_time=0.4, score=0.9)
    assert _alignment_item_to_word(item) == {
        "word": "hello",
        "start": 0.1,
        "end": 0.4,
        "score": 0.9,
    }

    assert _alignment_item_to_word({"word": "world", "start": 0.5, "end": 0.8}) == {
        "word": "world",
        "start": 0.5,
        "end": 0.8,
    }


def test_coalesce_text_segments_respects_max_chunk_seconds() -> None:
    chunks = _coalesce_text_segments(
        [
            {"start": 0.0, "end": 10.0, "text": "first"},
            {"start": 10.0, "end": 20.0, "text": "second"},
            {"start": 20.0, "end": 35.0, "text": "third"},
        ],
        max_chunk_seconds=25,
    )

    assert chunks == [
        {"start": 0.0, "end": 20.0, "text": "first second"},
        {"start": 20.0, "end": 35.0, "text": "third"},
    ]


def test_offset_words_maps_chunk_words_back_to_original_timeline() -> None:
    assert _offset_words([{"word": "hello", "start": 0.2, "end": 0.5}], offset=10.0) == [
        {"word": "hello", "start": 10.2, "end": 10.5}
    ]


class _FakeAligner:
    """Sleeps past the idle timeout and calls `during_align` on the first batch."""

    def __init__(self, during_align=None) -> None:
        self.during_align = during_align
        self.batches = 0

    def align(self, *, audio, text, language):
        self.batches += 1
        if self.batches == 1 and self.during_align is not None:
            self.during_align()
        time.sleep(0.05)
        return [[{"text": t, "start_time": 0.0, "end_time": 0.5}] for t in text]


def _alignment_manager(monkeypatch, aligner: _FakeAligner) -> tuple[ForcedAlignmentModelManager, list[bool]]:
    from parakeetx_api_server.model_managers import forced_alignment_manager as module

    monkeypatch.setitem(
        sys.modules,
        "qwen_asr",
        SimpleNamespace(Qwen3ForcedAligner=SimpleNamespace(from_pretrained=lambda *_a, **_k: aligner)),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))
    released: list[bool] = []
    monkeypatch.setattr(module, "release_memory_to_os", lambda *, clear_cuda=False: released.append(clear_cuda))
    settings = ForcedAlignmentSettings(device="cuda", max_chunk_seconds=1.0, batch_size=1)
    # 0.0001 min = 6 ms, well under the fake aligner's per-batch sleep.
    return ForcedAlignmentModelManager(settings, idle_evict_minutes=0.0001), released


def _two_chunk_audio(tmp_path: Path) -> tuple[Path, list[dict]]:
    audio_path = tmp_path / "audio.wav"
    sf.write(str(audio_path), np.zeros(3 * 16000, dtype=np.float32), 16000)
    segments = [{"start": 0.0, "end": 1.0, "text": "one"}, {"start": 2.0, "end": 3.0, "text": "two"}]
    return audio_path, segments


def test_forced_aligner_is_not_evicted_during_alignment(monkeypatch, tmp_path: Path) -> None:
    aligner = _FakeAligner()
    manager, released = _alignment_manager(monkeypatch, aligner)
    audio_path, segments = _two_chunk_audio(tmp_path)
    loaded_during_align: list[bool] = []
    aligner.during_align = lambda: (time.sleep(0.05), loaded_during_align.append(manager.status()["loaded"]))

    words = manager.align_segments(audio_path, segments=segments, language="en")

    assert [w["word"] for w in words] == ["one", "two"]
    assert loaded_during_align == [True]
    deadline = time.monotonic() + 1.0
    while manager.status()["loaded"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.status()["loaded"] is False
    assert released == [True]


def test_forced_alignment_survives_unload_mid_request(monkeypatch, tmp_path: Path) -> None:
    aligner = _FakeAligner()
    manager, _ = _alignment_manager(monkeypatch, aligner)
    aligner.during_align = manager.unload_model
    audio_path, segments = _two_chunk_audio(tmp_path)

    words = manager.align_segments(audio_path, segments=segments, language="en")

    assert [w["word"] for w in words] == ["one", "two"]
    assert aligner.batches == 2
