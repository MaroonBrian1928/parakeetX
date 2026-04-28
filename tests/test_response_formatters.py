from __future__ import annotations

from parakeetx_api_server.services.response_formatters import as_srt, as_verbose_json, as_vtt


def _payload():
    return {
        "text": "hello world",
        "language": "en",
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.2,
                "text": "hello world",
                "speaker": "SPEAKER_00",
            }
        ],
        "words": [
            {
                "word": "hello",
                "start": 0.0,
                "end": 0.5,
                "speaker": "SPEAKER_00",
            }
        ],
        "diarization": [{"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"}],
    }


def test_verbose_json_contains_diarization_and_speakers():
    verbose = as_verbose_json(_payload())
    assert verbose["segments"][0]["speaker"] == "SPEAKER_00"
    assert verbose["words"][0]["speaker"] == "SPEAKER_00"
    assert verbose["diarization"][0]["speaker"] == "SPEAKER_00"


def test_srt_and_vtt_prefix_speaker_labels():
    srt = as_srt(_payload())
    vtt = as_vtt(_payload())

    assert "[SPEAKER_00] hello world" in srt
    assert "WEBVTT" in vtt
    assert "[SPEAKER_00] hello world" in vtt
