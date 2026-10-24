from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any


def as_text(payload: dict[str, Any]) -> str:
    return str(payload.get("text", "")).strip()


def as_json(payload: dict[str, Any]) -> dict[str, str]:
    return {"text": str(payload.get("text", ""))}


def as_verbose_json(payload: dict[str, Any]) -> dict[str, Any]:
    word_segments = [_as_whisperx_word(word) for word in payload.get("words", [])]
    segments = _segments_with_words(payload.get("segments", []), word_segments)

    return {
        "task": "transcribe",
        "language": payload.get("language", "en"),
        "duration": _duration(segments),
        "text": payload.get("text", ""),
        "words": word_segments,
        "word_segments": word_segments,
        "segments": segments,
        "diarization": payload.get("diarization", []),
        "vad": payload.get("vad", []),
        "model": payload.get("model"),
    }


def as_srt(payload: dict[str, Any]) -> str:
    return _subtitle(payload.get("segments", []), vtt=False)


def as_vtt(payload: dict[str, Any]) -> str:
    content = _subtitle(payload.get("segments", []), vtt=True)
    return f"WEBVTT\n\n{content}" if content else "WEBVTT\n"


def _duration(segments: list[dict[str, Any]]) -> float:
    if not segments:
        return 0.0
    return float(max(float(s.get("end", 0.0)) for s in segments))


def _segments_with_words(
    segments: list[dict[str, Any]],
    word_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach each word to every segment whose [start, end] contains its midpoint.

    A word whose midpoint falls in no segment goes to the nearest one instead (the earlier
    on a tie), so ASR words in the gap between segments are never dropped.
    """
    order = sorted(range(len(word_segments)), key=lambda i: _word_midpoint(word_segments[i]))
    midpoints = [_word_midpoint(word_segments[i]) for i in order]
    members: list[list[int]] = []
    placed = [False] * len(word_segments)

    for segment in segments:
        lo = bisect_left(midpoints, float(segment.get("start", 0.0)))
        hi = bisect_right(midpoints, float(segment.get("end", 0.0)))
        members.append(order[lo:hi])
        for i in order[lo:hi]:
            placed[i] = True

    candidates = [
        (float(s.get("start", 0.0)), float(s.get("end", 0.0)), index)
        for index, s in enumerate(segments)
        if float(s.get("end", 0.0)) >= float(s.get("start", 0.0))
    ]
    if candidates:
        by_end = sorted(candidates, key=lambda c: (c[1], -c[2]))
        ends = [c[1] for c in by_end]
        by_start = sorted(candidates, key=lambda c: (c[0], c[2]))
        starts = [c[0] for c in by_start]
        for i in (i for i in order if not placed[i]):
            midpoint = _word_midpoint(word_segments[i])
            before = bisect_left(ends, midpoint) - 1
            after = bisect_right(starts, midpoint)
            nearest = None
            if before >= 0:
                nearest = (midpoint - by_end[before][1], by_end[before][2])
            if after < len(by_start):
                distance = by_start[after][0] - midpoint
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, by_start[after][2])
            if nearest is not None:
                members[nearest[1]].append(i)

    output: list[dict[str, Any]] = []
    for segment, indices in zip(segments, members):
        output_segment = dict(segment)
        output_segment["words"] = [word_segments[i] for i in sorted(indices)]
        output.append(output_segment)

    return output


def _word_midpoint(word: dict[str, Any]) -> float:
    word_start = float(word.get("start", 0.0))
    word_end = float(word.get("end", word_start))
    return word_start + max(0.0, word_end - word_start) / 2.0


def _as_whisperx_word(word: dict[str, Any]) -> dict[str, Any]:
    output = {
        "word": word.get("word", ""),
        "start": word.get("start", 0.0),
        "end": word.get("end", 0.0),
    }

    score = word.get("score", word.get("confidence"))
    if score is not None:
        output["score"] = score

    speaker = word.get("speaker")
    if speaker:
        output["speaker"] = speaker

    return output


def _subtitle(segments: list[dict[str, Any]], *, vtt: bool) -> str:
    lines: list[str] = []
    for i, segment in enumerate(segments, start=1):
        start = _format_ts(float(segment.get("start", 0.0)), vtt=vtt)
        end = _format_ts(float(segment.get("end", 0.0)), vtt=vtt)

        text = str(segment.get("text", "")).strip()
        speaker = segment.get("speaker")
        if speaker:
            text = f"[{speaker}] {text}".strip()

        if not vtt:
            lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + ("\n" if lines else "")


def _format_ts(seconds: float, *, vtt: bool) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    sep = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"
