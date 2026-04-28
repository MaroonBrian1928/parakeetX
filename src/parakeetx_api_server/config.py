from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ParakeetSettings(BaseModel):
    model_name: str = "nvidia/parakeet-tdt-0.6b-v2"
    device: str = "cpu"
    preload_model: bool = False
    local_files_only: bool = False
    cuda_half_precision: bool = True
    cuda_adaptive_chunking: bool = True
    cuda_chunk_seconds_override: int | None = Field(default=None, ge=1)
    cuda_chunk_min_seconds: int = Field(default=30, ge=1)
    cuda_chunk_max_seconds: int = Field(default=1200, ge=1)
    cuda_chunk_overlap_seconds: float = Field(default=0.0, ge=0.0, le=10.0)


class DiarizationSettings(BaseModel):
    model_name: str = "pyannote/speaker-diarization-community-1"
    device: str = "cpu"
    preload_model: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    api_key: str | None = None
    hf_token: str | None = None

    parakeet: ParakeetSettings = Field(default_factory=ParakeetSettings)
    diarization: DiarizationSettings = Field(default_factory=DiarizationSettings)

    max_concurrent_transcriptions: int = 2
    debug_log_transcription_payload: bool = False
    model_idle_evict_minutes: float | None = Field(default=None, ge=0)

    uvicorn_host: str = "0.0.0.0"
    uvicorn_port: int = 7317

    @field_validator("model_idle_evict_minutes", mode="before")
    @classmethod
    def _empty_idle_evict_minutes_as_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    def configured_api_keys(self) -> set[str]:
        keys: set[str] = set()
        if self.api_key:
            keys.add(self.api_key.strip())

        return {key for key in keys if key}


@lru_cache
def get_settings() -> Settings:
    return Settings()
