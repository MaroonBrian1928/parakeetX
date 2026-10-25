"""Compare native Parakeet word timings with Qwen forced alignment and the merged result on one audio file.

Runs inside a ParakeetX image with the forced-alignment extra installed, e.g.:

    docker cp scripts/bench_alignment.py <container>:/tmp/
    docker cp episode.mp3 <container>:/tmp/
    docker exec -w /tmp <container> python /tmp/bench_alignment.py /tmp/episode.mp3 --start 400 --duration 360

A segment counts as failed when any of its aligned words is zero-length, starts before the
previous word, or sits more than --tolerance seconds outside the segment. The merged timeline is
what the server returns: Qwen timings with native fallback per word (merge_aligned_words).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from parakeetx_api_server.config import get_settings
from parakeetx_api_server.model_managers.diarization_manager import DiarizationModelManager
from parakeetx_api_server.model_managers.forced_alignment_manager import (
    ForcedAlignmentModelManager,
    merge_aligned_words,
)
from parakeetx_api_server.model_managers.parakeet_manager import ParakeetModelManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="clip start in seconds")
    parser.add_argument("--duration", type=float, default=None, help="clip length in seconds (default: to the end)")
    parser.add_argument("--tolerance", type=float, default=0.25, help="seconds a word may sit outside its segment")
    parser.add_argument(
        "--diarize", action="store_true", help="split failures by whether the segment has overlapping speakers"
    )
    parser.add_argument("--dump", type=Path, help="write segments and the native, qwen and merged word timelines as JSON")
    parser.add_argument("--show", type=float, nargs=2, metavar=("FROM", "TO"), help="print both timings in this window")
    args = parser.parse_args()

    settings = get_settings()
    with tempfile.TemporaryDirectory() as tmpdir:
        wav = Path(tmpdir) / "clip.wav"
        command = ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(args.start)]
        if args.duration is not None:
            command += ["-t", str(args.duration)]
        subprocess.run([*command, "-i", str(args.audio), "-ac", "1", "-ar", "16000", str(wav)], check=True)

        started = time.perf_counter()
        asr = ParakeetModelManager(settings.parakeet).transcribe(wav)
        asr_seconds = time.perf_counter() - started

        started = time.perf_counter()
        aligned = ForcedAlignmentModelManager(settings.forced_alignment).align_segments(
            wav, segments=asr["segments"], language="en"
        )
        align_seconds = time.perf_counter() - started

        diarization = (
            DiarizationModelManager(settings.diarization, settings.hf_token).diarize(wav) if args.diarize else None
        )

    native = asr["words"]
    segments = asr["segments"]
    merged, fallback_words = merge_aligned_words(native, aligned, segments)
    pairs = _pair_words(native, aligned)
    groups = _group_by_segment(native, segments)
    failures = [
        _failure_reasons([aligned[pairs[i]] if i in pairs else None for i in indices], start, end, args.tolerance)
        for start, end, indices in groups
    ]
    failed = [reasons for reasons in failures if reasons]
    diffs = sorted(abs(aligned[j]["start"] - native[i]["start"]) for i, j in pairs.items())

    print(f"clip: {args.audio} start={args.start}s duration={args.duration or 'full'}")
    print(f"timing: asr={asr_seconds:.2f}s qwen_alignment={align_seconds:.2f}s")
    print(f"segments={len(segments)} words native={len(native)} aligned={len(aligned)} paired={len(pairs)}")
    print(
        f"zero-length words: native={_zero_length(native)} qwen={_zero_length(aligned)} merged={_zero_length(merged)}"
    )
    print(f"merged words using native timing: {fallback_words}/{len(merged)}")
    print(f"segments with failed qwen alignment: {len(failed)} ({100 * len(failed) / max(1, len(segments)):.1f}%)")
    for reason in ("unpaired", "zero-length", "out of order", "outside segment"):
        print(f"  {reason}: {sum(reason in r for r in failed)}")
    if diffs:
        print(
            f"|qwen - native| start difference: median={diffs[len(diffs) // 2]:.3f}s "
            f"p90={diffs[int(len(diffs) * 0.9)]:.3f}s"
        )

    if diarization is not None:
        overlaps = _overlap_regions(diarization)
        in_overlap = [_intersects(start, end, overlaps) for start, end, _ in groups]
        failed_in_overlap = sum(1 for reasons, hit in zip(failures, in_overlap) if reasons and hit)
        print(
            f"overlapping speech: {sum(e - s for s, e in overlaps):.1f}s in {len(overlaps)} regions; "
            f"{sum(in_overlap)}/{len(groups)} segments ({100 * sum(in_overlap) / max(1, len(groups)):.1f}%) touch it"
        )
        print(
            f"  failed segments touching overlap: {failed_in_overlap}/{len(failed)}; "
            f"failure rate with overlap {_rate(failures, in_overlap, True)}, without {_rate(failures, in_overlap, False)}"
        )

    if args.dump:
        args.dump.write_text(
            json.dumps({"start": args.start, "segments": segments, "native": native, "qwen": aligned, "merged": merged}),
            encoding="utf-8",
        )

    if args.show:
        low, high = args.show
        for label, words in (("native", native), ("qwen", aligned), ("merged", merged)):
            window = [w for w in words if low <= w["start"] + args.start <= high]
            print(f"\n{label}:")
            print(" ".join(f'{w["word"]}@{w["start"] + args.start:.2f}-{w["end"] + args.start:.2f}' for w in window))


def _normalize(word: Any) -> str:
    return re.sub(r"[^\w']", "", str(word).lower().replace("’", "'"))


def _pair_words(native: list[dict[str, Any]], aligned: list[dict[str, Any]]) -> dict[int, int]:
    aligned_tokens = [_normalize(w["word"]) for w in aligned]
    pairs: dict[int, int] = {}
    cursor = 0
    for i, word in enumerate(native):
        token = _normalize(word["word"])
        for k in range(cursor, min(cursor + 4, len(aligned_tokens))):
            if token and aligned_tokens[k] == token:
                pairs[i] = k
                cursor = k + 1
                break
    return pairs


def _group_by_segment(
    native: list[dict[str, Any]], segments: list[dict[str, Any]]
) -> list[tuple[float, float, list[int]]]:
    groups: list[tuple[float, float, list[int]]] = []
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        indices = [i for i, w in enumerate(native) if start <= (w["start"] + w["end"]) / 2 <= end]
        if indices:
            groups.append((start, end, indices))
    return groups


def _failure_reasons(
    words: list[dict[str, Any] | None], seg_start: float, seg_end: float, tolerance: float
) -> set[str]:
    reasons: set[str] = set()
    previous_start = float("-inf")
    for word in words:
        if word is None:
            reasons.add("unpaired")
            continue
        if word["end"] <= word["start"]:
            reasons.add("zero-length")
        if word["start"] < previous_start:
            reasons.add("out of order")
        if word["start"] < seg_start - tolerance or word["end"] > seg_end + tolerance:
            reasons.add("outside segment")
        previous_start = word["start"]
    return reasons


def _overlap_regions(diarization: list[dict[str, Any]]) -> list[tuple[float, float]]:
    events = sorted(
        [(float(d["start"]), 1) for d in diarization] + [(float(d["end"]), -1) for d in diarization],
        key=lambda event: (event[0], event[1]),
    )
    regions: list[tuple[float, float]] = []
    active = 0
    opened = 0.0
    for at, delta in events:
        if active < 2 <= active + delta:
            opened = at
        elif active >= 2 > active + delta and at > opened:
            regions.append((opened, at))
        active += delta
    return regions


def _intersects(start: float, end: float, regions: list[tuple[float, float]], minimum: float = 0.1) -> bool:
    return any(min(end, r_end) - max(start, r_start) >= minimum for r_start, r_end in regions)


def _rate(failures: list[set[str]], flags: list[bool], want: bool) -> str:
    chosen = [bool(reasons) for reasons, flag in zip(failures, flags) if flag == want]
    return f"{sum(chosen)}/{len(chosen)} ({100 * sum(chosen) / max(1, len(chosen)):.0f}%)"


def _zero_length(words: list[dict[str, Any]]) -> int:
    return sum(1 for w in words if w["end"] <= w["start"])


if __name__ == "__main__":
    main()
