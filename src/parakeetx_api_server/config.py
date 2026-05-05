from __future__ import annotations

from functools import lru_cache

from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _strip_env_string(value: object) -> object:
    if isinstance(value, str):
        return value.strip()
    return value


def _none_if_blank_env(value: object) -> object:
    value = _strip_env_string(value)
    if isinstance(value, str) and value.lower() in {"", "none", "null", "unset"}:
        return None
    return value


class ParakeetSettings(BaseModel):
    model_name: str = "cstr/parakeet-tdt-0.6b-v3-GGUF"
    model_path: Path = Path("/models/parakeet-tdt-0.6b-v3-q8_0.gguf")

    @field_validator("*", mode="before")
    @classmethod
    def _strip_string_values(cls, value: object) -> object:
        return _strip_env_string(value)


class DiarizationSettings(BaseModel):
    model_name: str = "pyannote/speaker-diarization-community-1"
    device: str = "cpu"
    preload_model: bool = False

    @field_validator("*", mode="before")
    @classmethod
    def _strip_string_values(cls, value: object) -> object:
        return _strip_env_string(value)


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
    uvicorn_port: int = 7474

    @field_validator("*", mode="before")
    @classmethod
    def _strip_string_values(cls, value: object) -> object:
        return _strip_env_string(value)

    @field_validator("api_key", "hf_token", "model_idle_evict_minutes", mode="before")
    @classmethod
    def _empty_optional_env_as_none(cls, value: object) -> object:
        return _none_if_blank_env(value)

    def configured_api_keys(self) -> set[str]:
        keys: set[str] = set()
        if self.api_key:
            keys.add(self.api_key.strip())

        return {key for key in keys if key}


@lru_cache
def get_settings() -> Settings:
    return Settings()
