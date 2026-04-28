from __future__ import annotations

from parakeetx_api_server.config import Settings


def test_parses_nested_config(monkeypatch):
    monkeypatch.setenv("API_KEY", "k1")
    monkeypatch.setenv("PARAKEET__MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2")
    monkeypatch.setenv("PARAKEET__DEVICE", "cuda")
    monkeypatch.setenv("DIARIZATION__PRELOAD_MODEL", "true")
    monkeypatch.setenv("MAX_CONCURRENT_TRANSCRIPTIONS", "7")

    settings = Settings()

    assert settings.parakeet.device == "cuda"
    assert settings.diarization.preload_model is True
    assert settings.max_concurrent_transcriptions == 7
    assert settings.configured_api_keys() == {"k1"}
