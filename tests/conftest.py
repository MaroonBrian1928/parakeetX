from __future__ import annotations

import io
import wave

import pytest
from fastapi.testclient import TestClient

from parakeetx_api_server.deps import (
    get_diarization_manager,
    get_parakeet_manager,
    get_transcription_service,
)
from parakeetx_api_server.main import app


@pytest.fixture(autouse=True)
def clear_dependency_caches(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("PARAKEET__MODEL_NAME", raising=False)
    monkeypatch.delenv("PARAKEET__DEVICE", raising=False)
    monkeypatch.delenv("PARAKEET__PRELOAD_MODEL", raising=False)
    monkeypatch.delenv("PARAKEET__LOCAL_FILES_ONLY", raising=False)
    monkeypatch.delenv("PARAKEET__CUDA_HALF_PRECISION", raising=False)
    monkeypatch.delenv("PARAKEET__CUDA_ADAPTIVE_CHUNKING", raising=False)
    monkeypatch.delenv("PARAKEET__CUDA_CHUNK_SECONDS_OVERRIDE", raising=False)
    monkeypatch.delenv("PARAKEET__CUDA_CHUNK_MIN_SECONDS", raising=False)
    monkeypatch.delenv("PARAKEET__CUDA_CHUNK_MAX_SECONDS", raising=False)
    monkeypatch.delenv("PARAKEET__CUDA_CHUNK_OVERLAP_SECONDS", raising=False)
    monkeypatch.delenv("DIARIZATION__MODEL_NAME", raising=False)
    monkeypatch.delenv("DIARIZATION__DEVICE", raising=False)
    monkeypatch.delenv("DIARIZATION__PRELOAD_MODEL", raising=False)
    monkeypatch.delenv("MAX_CONCURRENT_TRANSCRIPTIONS", raising=False)

    from parakeetx_api_server.config import get_settings

    get_settings.cache_clear()
    get_parakeet_manager.cache_clear()
    get_diarization_manager.cache_clear()
    get_transcription_service.cache_clear()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 16_000)
    return buffer.getvalue()
