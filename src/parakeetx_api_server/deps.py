from __future__ import annotations

from functools import lru_cache

from .config import Settings, get_settings
from .model_managers.diarization_manager import DiarizationModelManager
from .model_managers.model_worker import ModelWorkerClient
from .model_managers.parakeet_manager import ParakeetModelManager
from .model_managers.vad_manager import VadModelManager
from .services.transcription import TranscriptionService


@lru_cache
def get_model_worker_client() -> ModelWorkerClient | None:
    settings = get_settings()
    if not settings.model_process_isolation:
        return None
    return ModelWorkerClient(settings)


@lru_cache
def get_parakeet_manager() -> ParakeetModelManager:
    settings = get_settings()
    return ParakeetModelManager(
        settings.parakeet,
        idle_evict_minutes=settings.model_idle_evict_minutes,
        worker_client=get_model_worker_client(),
    )


@lru_cache
def get_diarization_manager() -> DiarizationModelManager:
    settings = get_settings()
    return DiarizationModelManager(
        settings.diarization,
        settings.hf_token,
        idle_evict_minutes=settings.model_idle_evict_minutes,
        worker_client=get_model_worker_client(),
    )


@lru_cache
def get_vad_manager() -> VadModelManager:
    settings = get_settings()
    return VadModelManager(settings.vad)


@lru_cache
def get_transcription_service() -> TranscriptionService:
    settings: Settings = get_settings()
    return TranscriptionService(
        parakeet_manager=get_parakeet_manager(),
        diarization_manager=get_diarization_manager(),
        vad_manager=get_vad_manager(),
        max_concurrency=settings.max_concurrent_transcriptions,
        unload_asr_before_diarization=settings.unload_asr_before_diarization,
    )
