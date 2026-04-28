from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16_000


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)

    if audio.size == 0:
        return audio.astype(np.float32)

    duration = audio.shape[0] / float(source_rate)
    output_length = max(1, int(round(duration * target_rate)))
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=output_length, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def normalize_audio_to_wav(input_path: Path, output_path: Path) -> Path:
    data, sample_rate = sf.read(str(input_path), always_2d=True)

    mono = data.mean(axis=1)
    mono = _resample_linear(mono, int(sample_rate), TARGET_SAMPLE_RATE)
    mono = np.clip(mono, -1.0, 1.0)

    sf.write(
        str(output_path),
        mono,
        TARGET_SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )

    return output_path
