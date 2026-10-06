"""The sweep-based speaker assignment and word grouping must match the original
quadratic scans exactly, including ties and boundary midpoints."""

import copy
import random

from parakeetx_api_server.services.response_formatters import _segments_with_words
from parakeetx_api_server.services.speaker_assignment import assign_speakers


def _reference_best_speaker(start, end, diarization_segments):
    best_label, best_score = None, 0.0
    for segment in diarization_segments:
        score = max(0.0, min(end, float(segment["end"])) - max(start, float(segment["start"])))
        if score > best_score:
            best_score, best_label = score, str(segment["speaker"])
    return best_label


def _reference_assign(items, diarization_segments):
    for item in items:
        speaker = _reference_best_speaker(
            float(item.get("start", 0.0)), float(item.get("end", 0.0)), diarization_segments
        )
        if speaker is not None:
            item["speaker"] = speaker


def _reference_segments_with_words(segments, words):
    output = []
    for segment in segments:
        seg_start, seg_end = float(segment.get("start", 0.0)), float(segment.get("end", 0.0))
        members = []
        for word in words:
            w_start = float(word.get("start", 0.0))
            w_end = float(word.get("end", w_start))
            mid = w_start + max(0.0, w_end - w_start) / 2.0
            if seg_start <= mid <= seg_end:
                members.append(word)
        output.append({**segment, "words": members})
    return output


def _spans(rng, count, *, max_len, prefix):
    # Coarse grid so ties, shared boundaries and zero/inverted spans actually occur.
    spans = []
    for i in range(count):
        start = rng.randint(0, 60) * 0.5
        end = start + rng.randint(-1, max_len) * 0.5
        spans.append({"start": start, "end": end, prefix: f"{prefix}{i % 4}"})
    if rng.random() < 0.5:
        rng.shuffle(spans)
    return spans


def test_assign_speakers_matches_reference() -> None:
    rng = random.Random(1234)
    for _ in range(500):
        diarization = _spans(rng, rng.randint(0, 25), max_len=12, prefix="speaker")
        words = _spans(rng, rng.randint(0, 40), max_len=3, prefix="word")
        segments = _spans(rng, rng.randint(0, 10), max_len=10, prefix="text")

        expected_words, expected_segments = copy.deepcopy(words), copy.deepcopy(segments)
        _reference_assign(expected_words, diarization)
        _reference_assign(expected_segments, diarization)

        assert assign_speakers(words, segments, diarization) == (expected_words, expected_segments)


def test_segments_with_words_matches_reference() -> None:
    rng = random.Random(5678)
    for _ in range(500):
        words = _spans(rng, rng.randint(0, 40), max_len=3, prefix="word")
        segments = _spans(rng, rng.randint(0, 10), max_len=10, prefix="text")

        assert _segments_with_words(segments, words) == _reference_segments_with_words(segments, words)
