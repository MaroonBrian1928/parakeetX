from __future__ import annotations

from parakeetx_api_server.config import Settings


def test_parses_nested_config(monkeypatch):
    monkeypatch.setenv("API_KEY", "k1")
    monkeypatch.setenv("PARAKEET__MODEL_PATH", "/tmp/parakeet.gguf")
    monkeypatch.setenv("DIARIZATION__PRELOAD_MODEL", "true")
    monkeypatch.setenv("MAX_CONCURRENT_TRANSCRIPTIONS", "7")
    monkeypatch.setenv("MODEL_IDLE_EVICT_MINUTES", "15")

    settings = Settings()

    assert settings.parakeet.model_name == "cstr/parakeet-tdt-0.6b-v3-GGUF"
    assert str(settings.parakeet.model_path) == "/tmp/parakeet.gguf"
    assert settings.diarization.preload_model is True
    assert settings.max_concurrent_transcriptions == 7
    assert settings.model_idle_evict_minutes == 15
    assert settings.configured_api_keys() == {"k1"}


def test_blank_optional_numeric_env_values_parse_as_none(monkeypatch):
    monkeypatch.setenv("MODEL_IDLE_EVICT_MINUTES", "")

    settings = Settings()

    assert settings.model_idle_evict_minutes is None


def test_optional_env_values_accept_common_unset_markers(monkeypatch):
    monkeypatch.setenv("API_KEY", " none ")
    monkeypatch.setenv("HF_TOKEN", " null ")
    monkeypatch.setenv("MODEL_IDLE_EVICT_MINUTES", " unset ")

    settings = Settings()

    assert settings.api_key is None
    assert settings.hf_token is None
    assert settings.model_idle_evict_minutes is None
    assert settings.configured_api_keys() == set()


def test_env_values_are_trimmed_before_parsing(monkeypatch):
    monkeypatch.setenv("API_KEY", "  k1  ")
    monkeypatch.setenv("PARAKEET__MODEL_PATH", " /tmp/model.gguf ")
    monkeypatch.setenv("DIARIZATION__DEVICE", " cpu ")
    monkeypatch.setenv("MODEL_IDLE_EVICT_MINUTES", " 2.5 ")

    settings = Settings()

    assert settings.api_key == "k1"
    assert settings.configured_api_keys() == {"k1"}
    assert str(settings.parakeet.model_path) == "/tmp/model.gguf"
    assert settings.diarization.device == "cpu"
    assert settings.model_idle_evict_minutes == 2.5


def test_removed_asr_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("PARAKEET__CUDA_CHUNK_SECONDS_OVERRIDE", "120")
    monkeypatch.setenv("PARAKEET__USE_EXTRACTED_NEMO_CACHE", "true")
    monkeypatch.setenv("PARAKEET__TORCH_LOAD_MMAP", "true")
    monkeypatch.setenv("MODEL_PROCESS_ISOLATION", "true")
    monkeypatch.setenv("UNLOAD_ASR_BEFORE_DIARIZATION", "true")

    settings = Settings()

    assert not hasattr(settings.parakeet, "cuda_chunk_seconds_override")
    assert not hasattr(settings.parakeet, "use_extracted_nemo_cache")
    assert not hasattr(settings.parakeet, "torch_load_mmap")
    assert not hasattr(settings, "model_process_isolation")
    assert not hasattr(settings, "unload_asr_before_diarization")
