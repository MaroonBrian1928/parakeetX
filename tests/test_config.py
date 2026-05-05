from __future__ import annotations

from parakeetx_api_server.config import Settings


def test_parses_nested_config(monkeypatch):
    monkeypatch.setenv("API_KEY", "k1")
    monkeypatch.setenv("PARAKEET__MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2")
    monkeypatch.setenv("PARAKEET__DEVICE", "cuda")
    monkeypatch.setenv("PARAKEET__CUDA_HALF_PRECISION", "true")
    monkeypatch.setenv("PARAKEET__CUDA_FORCE_GREEDY_DECODING", "true")
    monkeypatch.setenv("PARAKEET__CUDA_CHUNK_MIN_SECONDS", "45")
    monkeypatch.setenv("PARAKEET__USE_EXTRACTED_NEMO_CACHE", "false")
    monkeypatch.setenv("PARAKEET__TORCH_LOAD_MMAP", "false")
    monkeypatch.setenv("DIARIZATION__PRELOAD_MODEL", "true")
    monkeypatch.setenv("VAD__ENABLED", "true")
    monkeypatch.setenv("VAD__METHOD", " silero ")
    monkeypatch.setenv("VAD__USE_ONNX", "false")
    monkeypatch.setenv("VAD__ONNX_FALLBACK_TO_JIT", "false")
    monkeypatch.setenv("VAD__ONNX_OPSET_VERSION", "15")
    monkeypatch.setenv("VAD__VAD_ONSET", "0.4")
    monkeypatch.setenv("VAD__VAD_OFFSET", "0.25")
    monkeypatch.setenv("VAD__CHUNK_SIZE", "20")
    monkeypatch.setenv("MAX_CONCURRENT_TRANSCRIPTIONS", "7")
    monkeypatch.setenv("MODEL_IDLE_EVICT_MINUTES", "15")
    monkeypatch.setenv("MODEL_PROCESS_ISOLATION", "true")
    monkeypatch.setenv("UNLOAD_ASR_BEFORE_DIARIZATION", "false")

    settings = Settings()

    assert settings.parakeet.device == "cuda"
    assert settings.parakeet.cuda_half_precision is True
    assert settings.parakeet.cuda_force_greedy_decoding is True
    assert settings.parakeet.cuda_chunk_min_seconds == 45
    assert settings.parakeet.use_extracted_nemo_cache is False
    assert settings.parakeet.torch_load_mmap is False
    assert settings.diarization.preload_model is True
    assert settings.vad.enabled is True
    assert settings.vad.method == "silero"
    assert settings.vad.use_onnx is False
    assert settings.vad.onnx_fallback_to_jit is False
    assert settings.vad.onnx_opset_version == 15
    assert settings.vad.vad_onset == 0.4
    assert settings.vad.vad_offset == 0.25
    assert settings.vad.chunk_size == 20
    assert settings.max_concurrent_transcriptions == 7
    assert settings.model_idle_evict_minutes == 15
    assert settings.model_process_isolation is True
    assert settings.unload_asr_before_diarization is False
    assert settings.configured_api_keys() == {"k1"}


def test_blank_optional_numeric_env_values_parse_as_none(monkeypatch):
    monkeypatch.setenv("PARAKEET__CUDA_CHUNK_SECONDS_OVERRIDE", "")
    monkeypatch.setenv("MODEL_IDLE_EVICT_MINUTES", "")

    settings = Settings()

    assert settings.parakeet.cuda_chunk_seconds_override is None
    assert settings.model_idle_evict_minutes is None


def test_optional_env_values_accept_common_unset_markers(monkeypatch):
    monkeypatch.setenv("API_KEY", " none ")
    monkeypatch.setenv("HF_TOKEN", " null ")
    monkeypatch.setenv("PARAKEET__CUDA_CHUNK_SECONDS_OVERRIDE", " unset ")
    monkeypatch.setenv("MODEL_IDLE_EVICT_MINUTES", " unset ")

    settings = Settings()

    assert settings.api_key is None
    assert settings.hf_token is None
    assert settings.parakeet.cuda_chunk_seconds_override is None
    assert settings.model_idle_evict_minutes is None
    assert settings.configured_api_keys() == set()


def test_env_values_are_trimmed_before_parsing(monkeypatch):
    monkeypatch.setenv("API_KEY", "  k1  ")
    monkeypatch.setenv("PARAKEET__DEVICE", " cuda ")
    monkeypatch.setenv("PARAKEET__CUDA_HALF_PRECISION", " true ")
    monkeypatch.setenv("PARAKEET__CUDA_CHUNK_SECONDS_OVERRIDE", " 0 ")
    monkeypatch.setenv("DIARIZATION__DEVICE", " cpu ")
    monkeypatch.setenv("MODEL_IDLE_EVICT_MINUTES", " 2.5 ")

    settings = Settings()

    assert settings.api_key == "k1"
    assert settings.configured_api_keys() == {"k1"}
    assert settings.parakeet.device == "cuda"
    assert settings.parakeet.cuda_half_precision is True
    assert settings.parakeet.cuda_chunk_seconds_override is None
    assert settings.diarization.device == "cpu"
    assert settings.model_idle_evict_minutes == 2.5
