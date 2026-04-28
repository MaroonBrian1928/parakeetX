from __future__ import annotations

import asyncio
import logging
import sys
import time
import warnings

from fastapi import FastAPI, Request

from .config import get_settings
from .deps import (
    get_diarization_manager,
    get_parakeet_manager,
)
from .log_filters import install_noisy_dependency_log_filters
from .routers import models, transcriptions, translations

app = FastAPI(
    title="ParakeetX API Server",
    version="0.1.0",
    description="OpenAI-compatible transcription API powered by NVIDIA Parakeet and optional pyannote diarization.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.middleware("http")
async def log_transcription_request_wall_time(request: Request, call_next):
    if request.url.path != "/v1/audio/transcriptions":
        return await call_next(request)

    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        _emit_transcription_http_timing(
            started_at,
            status="error",
            extra=f"method={request.method} error={type(exc).__name__}",
        )
        raise

    _emit_transcription_http_timing(
        started_at,
        status=str(response.status_code),
        extra=f"method={request.method}",
    )
    return response


@app.on_event("startup")
async def startup() -> None:
    # Attach application logs to uvicorn's visible handlers so per-request
    # ASR chunk planning and model-routing diagnostics appear in container logs.
    app_logger = logging.getLogger("parakeetx_api_server")
    app_logger.handlers = list(logging.getLogger("uvicorn.error").handlers)
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False
    install_noisy_dependency_log_filters()

    # NeMo pulls pydub during model initialization; suppress noisy upstream
    # SyntaxWarning lines from pydub regex strings without muting other warnings.
    warnings.filterwarnings(
        "ignore",
        category=SyntaxWarning,
        module=r"pydub\.utils",
    )

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


def _emit_transcription_http_timing(
    started_at: float,
    *,
    status: str,
    extra: str,
) -> None:
    elapsed = time.perf_counter() - started_at
    print(
        (
            "Transcription timing: stage=http_request "
            f"elapsed={elapsed:.2f}s total={elapsed:.2f}s status={status} {extra}"
        ),
        file=sys.stderr,
        flush=True,
    )
