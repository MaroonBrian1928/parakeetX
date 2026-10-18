from __future__ import annotations

from typing import Any


def _assign_best_speaker(
    items: list[dict[str, Any]],
    diarization_segments: list[dict[str, Any]],
) -> None:
    """Label each item with the speaker whose segment overlaps it most.

    Sweeps items and segments in time order so each item only compares against
    segments still active around it. Ties go to the earliest segment in
    ``diarization_segments`` order.
    """
    segments = sorted(
        (
            (float(s["start"]), float(s["end"]), index, str(s["speaker"]))
            for index, s in enumerate(diarization_segments)
        ),
        key=lambda s: s[0],
    )
    order = sorted(range(len(items)), key=lambda i: float(items[i].get("start", 0.0)))

    active: list[tuple[float, float, int, str]] = []
    next_segment = 0
    for i in order:
        item = items[i]
        start = float(item.get("start", 0.0))
        end = float(item.get("end", 0.0))

        while next_segment < len(segments) and segments[next_segment][0] < end:
            active.append(segments[next_segment])
            next_segment += 1
        active = [s for s in active if s[1] > start]

        best: tuple[float, int, str] | None = None
        for seg_start, seg_end, index, speaker in active:
            score = min(end, seg_end) - max(start, seg_start)
            if score > 0 and (best is None or score > best[0] or (score == best[0] and index < best[1])):
                best = (score, index, speaker)

        if best is not None:
            item["speaker"] = best[2]


def assign_speakers(
    words: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    diarization_segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _assign_best_speaker(words, diarization_segments)
    _assign_best_speaker(segments, diarization_segments)
    return words, segments
