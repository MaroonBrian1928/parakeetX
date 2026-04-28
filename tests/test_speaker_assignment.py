from __future__ import annotations

from parakeetx_api_server.services.speaker_assignment import assign_speakers


def test_assigns_speaker_by_max_overlap():
    words = [
        {"word": "hi", "start": 0.0, "end": 0.4},
        {"word": "there", "start": 0.5, "end": 0.9},
    ]
    segments = [{"id": 0, "start": 0.0, "end": 0.9, "text": "hi there"}]
    diarization = [
        {"start": 0.0, "end": 0.45, "speaker": "SPEAKER_00"},
        {"start": 0.45, "end": 1.0, "speaker": "SPEAKER_01"},
    ]

    out_words, out_segments = assign_speakers(words, segments, diarization)

    assert out_words[0]["speaker"] == "SPEAKER_00"
    assert out_words[1]["speaker"] == "SPEAKER_01"
    assert out_segments[0]["speaker"] == "SPEAKER_00"
