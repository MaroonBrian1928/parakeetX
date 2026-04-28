from __future__ import annotations

from typing import Any


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _best_speaker(
    start: float,
    end: float,
    diarization_segments: list[dict[str, Any]],
) -> str | None:
    best_label: str | None = None
    best_score = 0.0

    for segment in diarization_segments:
        score = _overlap(start, end, float(segment["start"]), float(segment["end"]))
        if score > best_score:
            best_score = score
            best_label = str(segment["speaker"])

    return best_label


def assign_speakers(
    words: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    diarization_segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for word in words:
        speaker = _best_speaker(
            float(word.get("start", 0.0)),
            float(word.get("end", 0.0)),
            diarization_segments,
        )
        if speaker is not None:
            word["speaker"] = speaker

    for segment in segments:
        speaker = _best_speaker(
            float(segment.get("start", 0.0)),
            float(segment.get("end", 0.0)),
            diarization_segments,
        )
        if speaker is not None:
            segment["speaker"] = speaker

    return words, segments
