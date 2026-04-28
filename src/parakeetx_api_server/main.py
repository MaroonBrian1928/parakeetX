from __future__ import annotations

import asyncio

from fastapi import FastAPI

from .config import get_settings
from .deps import (
    get_diarization_manager,
    get_parakeet_manager,
)
from .routers import models, transcriptions, translations

app = FastAPI(
    title="ParakeetX API Server",
    version="0.1.0",
    description="OpenAI-compatible transcription API powered by NVIDIA Parakeet and optional pyannote diarization.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def startup() -> None:
    settings = get_settings()

    if settings.parakeet.preload_model:
        parakeet = get_parakeet_manager()
        await asyncio.to_thread(parakeet.load_model)

    if settings.diarization.preload_model:
        diarization = get_diarization_manager()
        await asyncio.to_thread(diarization.load_model)


app.include_router(transcriptions.router)
app.include_router(translations.router)
app.include_router(models.router)
