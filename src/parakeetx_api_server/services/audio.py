from __future__ import annotations

import subprocess
from pathlib import Path

TARGET_SAMPLE_RATE = 16_000


def normalize_audio_to_wav(input_path: Path, output_path: Path) -> Path:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-f",
        "wav",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to normalize audio for transcription") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = "ffmpeg failed to normalize audio"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from exc

    return output_path
