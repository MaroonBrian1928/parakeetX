from __future__ import annotations

from typing import Any


def as_text(payload: dict[str, Any]) -> str:
    return str(payload.get("text", "")).strip()


def as_json(payload: dict[str, Any]) -> dict[str, str]:
    return {"text": str(payload.get("text", ""))}


def as_verbose_json(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "transcribe",
        "language": payload.get("language", "en"),
        "duration": _duration(payload.get("segments", [])),
        "text": payload.get("text", ""),
        "words": payload.get("words", []),
        "segments": payload.get("segments", []),
        "diarization": payload.get("diarization", []),
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
