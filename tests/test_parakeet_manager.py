from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from parakeetx_api_server.config import ParakeetSettings
from parakeetx_api_server.model_managers import parakeet_manager as manager_module
from parakeetx_api_server.model_managers.parakeet_manager import (
    ParakeetModelManager,
    _build_crispasr_args,
    _normalize_crispasr_payload,
)


def test_build_crispasr_args_uses_canonical_parakeet_shape(tmp_path: Path) -> None:
    audio_path = tmp_path / "normalized.wav"
    output_base = tmp_path / "transcript"

    args = _build_crispasr_args(
        binary="crispasr",
        model_path=Path("/models/parakeet-tdt-0.6b-v3-q8_0.gguf"),
        audio_path=audio_path,
        output_base=output_base,
    )

    assert args == [
        "crispasr",
        "--backend",
        "parakeet",
        "-m",
        "/models/parakeet-tdt-0.6b-v3-q8_0.gguf",
        "-f",
        str(audio_path),
        "-ojf",
        "-of",
        str(output_base),
        "-np",
    ]


def test_build_crispasr_args_can_force_gpu_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CRISPASR_GPU_BACKEND", "cuda")

    args = _build_crispasr_args(
        binary="crispasr",
        model_path=Path("/models/model.gguf"),
        audio_path=tmp_path / "normalized.wav",
        output_base=tmp_path / "transcript",
    )

    assert args[-2:] == ["--gpu-backend", "cuda"]


def test_normalize_crispasr_payload_maps_words_segments_and_language() -> None:
    payload = _normalize_crispasr_payload(
        {
            "text": "hello world",
            "language": "en",
            "words": [
                {"word": "hello", "start_ms": 100, "end_ms": 400, "score": 0.91},
                {"text": "world", "start": "0.50s", "end": "900ms"},
            ],
            "segments": [
                {"start_ms": 100, "end_ms": 900, "text": "hello world"},
            ],
        },
        model_name="cstr/parakeet-tdt-0.6b-v3-GGUF",
    )

    assert payload == {
        "text": "hello world",
        "language": "en",
        "model": "cstr/parakeet-tdt-0.6b-v3-GGUF",
        "words": [
            {"word": "hello", "start": 0.1, "end": 0.4, "confidence": 0.91},
            {"word": "world", "start": 0.5, "end": 0.9, "confidence": None},
        ],
        "segments": [
            {"id": 0, "start": 0.1, "end": 0.9, "text": "hello world"},
        ],
    }


def test_normalize_crispasr_documented_transcription_layout() -> None:
    payload = _normalize_crispasr_payload(
        {
            "crispasr": {"backend": "parakeet", "language": "en"},
            "transcription": [
                {
                    "offsets": {"from": 240, "to": 10880},
                    "text": "hello world",
                    "words": [
                        {"word": "hello", "start_ms": 240, "end_ms": 640},
                    ],
                }
            ],
        },
        model_name="cstr/parakeet-tdt-0.6b-v3-GGUF",
    )

    assert payload["text"] == "hello world"
    assert payload["language"] == "en"
    assert payload["segments"] == [
        {"id": 0, "start": 0.24, "end": 10.88, "text": "hello world"}
    ]
    assert payload["words"] == [
        {"word": "hello", "start": 0.24, "end": 0.64, "confidence": None}
    ]


def test_load_model_validates_missing_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    monkeypatch.setattr(manager_module, "_resolve_binary", lambda binary: None)

    manager = ParakeetModelManager(ParakeetSettings(model_path=model_path))

    with pytest.raises(RuntimeError, match="CrispASR binary was not found"):
        manager.load_model()


def test_load_model_validates_missing_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(manager_module, "_resolve_binary", lambda binary: "/usr/bin/crispasr")

    manager = ParakeetModelManager(ParakeetSettings(model_path=tmp_path / "missing.gguf"))

    with pytest.raises(RuntimeError, match="Parakeet GGUF model file not found"):
        manager.load_model()


def test_transcribe_injects_mmap_env_and_reads_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.gguf"
    audio_path = tmp_path / "normalized.wav"
    model_path.write_bytes(b"gguf")
    audio_path.write_bytes(b"wav")
    seen_env = {}
    seen_args = []

    monkeypatch.setattr(manager_module, "_resolve_binary", lambda binary: "/usr/bin/crispasr")

    def fake_run(args, *, env, check, capture_output, text, timeout):
        _ = check
        _ = capture_output
        _ = text
        _ = timeout
        seen_args.extend(args)
        seen_env.update(env)
        output_base = Path(args[args.index("-of") + 1])
        output_base.with_suffix(".json").write_text(
            json.dumps({"text": "hello", "words": [], "segments": []}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(manager_module.subprocess, "run", fake_run)

    manager = ParakeetModelManager(ParakeetSettings(model_path=model_path))
    payload = manager.transcribe(audio_path)

    assert seen_env["CRISPASR_GGUF_MMAP"] == "1"
    assert "--backend" in seen_args
    assert "-ojf" in seen_args
    assert "-np" in seen_args
    assert str(model_path) in seen_args
    assert str(audio_path) in seen_args
    assert payload["text"] == "hello"


def test_transcribe_failure_includes_process_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.gguf"
    audio_path = tmp_path / "normalized.wav"
    model_path.write_bytes(b"gguf")
    audio_path.write_bytes(b"wav")
    monkeypatch.setattr(manager_module, "_resolve_binary", lambda binary: "/usr/bin/crispasr")

    def fake_run(args, **kwargs):
        _ = kwargs
        return subprocess.CompletedProcess(args, 2, stdout="out text", stderr="err text")

    monkeypatch.setattr(manager_module.subprocess, "run", fake_run)

    manager = ParakeetModelManager(ParakeetSettings(model_path=model_path))

    with pytest.raises(RuntimeError) as exc_info:
        manager.transcribe(audio_path)

    assert "CrispASR transcription failed" in str(exc_info.value)
    assert "out text" in str(exc_info.value)
    assert "err text" in str(exc_info.value)
