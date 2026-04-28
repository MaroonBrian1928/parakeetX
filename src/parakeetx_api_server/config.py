from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ParakeetSettings(BaseModel):
    model_name: str = "nvidia/parakeet-tdt-0.6b-v2"
    device: str = "cpu"
    preload_model: bool = False
    local_files_only: bool = False


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

    uvicorn_host: str = "0.0.0.0"
    uvicorn_port: int = 7317

    def configured_api_keys(self) -> set[str]:
        keys: set[str] = set()
        if self.api_key:
            keys.add(self.api_key.strip())

        return {key for key in keys if key}


@lru_cache
def get_settings() -> Settings:
    return Settings()
