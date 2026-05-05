from __future__ import annotations

import subprocess
from pathlib import Path

from parakeetx_api_server.services.audio import normalize_audio_to_wav


def test_normalize_audio_to_wav_invokes_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp3"
    output_path = tmp_path / "output.wav"
    input_path.write_bytes(b"fake")
    calls: list[list[str]] = []

    def fake_run(command, *, check, capture_output, text):
        calls.append(command)
        assert check is True
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert normalize_audio_to_wav(input_path, output_path) == output_path
    assert calls[0][:8] == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
    ]
    assert calls[0][-1] == str(output_path)
