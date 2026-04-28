from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse

from ..auth import require_api_key
from ..config import get_settings
from ..deps import get_transcription_service
from ..services.response_formatters import as_json, as_srt, as_text, as_verbose_json, as_vtt
from ..services.transcription import TranscriptionService

router = APIRouter(prefix="/v1/audio", tags=["audio"])
logger = logging.getLogger(__name__)

SUPPORTED_RESPONSE_FORMATS = {"json", "text", "srt", "vtt", "verbose_json", "diarized_json"}
SUPPORTED_TIMESTAMP_GRANULARITIES = {"word", "segment"}


def _friendly_runtime_error_detail(exc: RuntimeError) -> str:
    detail = str(exc).strip()
    lowered = detail.lower()

    if "out of memory" in lowered:
        return (
            f"{detail} "
            "Hint: GPU memory is too tight for this request. Try `diarize=false`, shorter audio, "
            "or set `PARAKEET__DEVICE_CUDA=cpu` (and optionally `DIARIZATION__DEVICE_CUDA=cpu`)."
        )

    if "invalid ptx" in lowered or "cuda error: invalid argument" in lowered:
        return (
            f"{detail} "
            "Hint: this GPU/runtime combo is incompatible with the current CUDA decoder path. "
            "Set `PARAKEET__DEVICE_CUDA=cpu` (and optionally `DIARIZATION__DEVICE_CUDA=cpu`) "
            "to run reliably on this host."
        )

    return detail


@router.post("/transcriptions", dependencies=[Depends(require_api_key)])
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    timestamp_granularities: Annotated[
        list[str] | None,
        Form(alias="timestamp_granularities[]"),
    ] = None,
    timestamp_granularities_plain: str | None = Form(default=None, alias="timestamp_granularities"),
    stream: bool = Form(default=False),
    diarize: bool = Form(default=False),
    min_speakers: int | None = Form(default=None),
    max_speakers: int | None = Form(default=None),
    num_speakers: int | None = Form(default=None),
    speaker_embeddings: bool = Form(default=False),
    highlight_words: bool = Form(default=False),
    prompt: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    hotwords: str | None = Form(default=None),
    forced_alignment: bool = Form(default=False),
    service: TranscriptionService = Depends(get_transcription_service),
):
    _ = speaker_embeddings
    _ = highlight_words

    resolved_model = getattr(service, "configured_model_name", None) or "nvidia/parakeet-tdt-0.6b-v2"
    if model and model != resolved_model:
        logger.info(
            "Ignoring client-provided model '%s'; using configured model '%s'.",
            model,
            resolved_model,
        )

    if stream:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Streaming responses are not supported.",
        )

    if language and language.lower() not in {"en", "english"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only English transcription is supported.",
        )

    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported response_format '{response_format}'.",
        )

    timestamps = list(timestamp_granularities or [])
    if timestamp_granularities_plain:
        plain = timestamp_granularities_plain.strip()
        parsed: list[str] = []
        if plain.startswith("["):
            try:
                loaded = json.loads(plain)
                if isinstance(loaded, list):
                    parsed = [str(item).strip() for item in loaded]
            except json.JSONDecodeError:
                parsed = []
        if not parsed:
            parsed = [item.strip() for item in plain.split(",") if item.strip()]
        timestamps.extend(parsed)

    if not timestamps:
        form = await request.form()
        parsed: list[str] = []
        for key, value in form.multi_items():
            if not key.startswith("timestamp_granularities"):
                continue
            if isinstance(value, str):
                parsed.append(value.strip())
        timestamps = [value for value in parsed if value]

    if any(granularity not in SUPPORTED_TIMESTAMP_GRANULARITIES for granularity in timestamps):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported timestamp_granularities[] value.",
        )

    settings = get_settings()
    if settings.debug_log_transcription_payload:
        logger.warning(
            "Incoming /v1/audio/transcriptions payload: %s",
            {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": getattr(file, "size", None),
                "model_requested": model,
                "model_used": resolved_model,
                "language": language,
                "response_format": response_format,
                "timestamp_granularities": timestamps,
                "stream": stream,
                "diarize": diarize,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "num_speakers": num_speakers,
                "speaker_embeddings": speaker_embeddings,
                "highlight_words": highlight_words,
                "prompt_present": bool(prompt),
                "temperature": temperature,
                "hotwords_present": bool(hotwords),
                "forced_alignment": forced_alignment,
            },
        )

    if prompt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prompt biasing is not supported.",
        )

    if temperature not in (None, 0, 0.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Temperature sampling is not supported.",
        )

    if hotwords:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Hotwords are not supported.",
        )

    if forced_alignment:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Forced alignment is not supported.",
        )

    try:
        payload = await service.transcribe_upload(
            upload=file,
            language=language,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_speakers=num_speakers,
        )
    except ValueError as exc:
        logger.warning("Transcription rejected: %s", exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Transcription runtime error")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_friendly_runtime_error_detail(exc),
        ) from exc

    payload["model"] = resolved_model

    if response_format == "text":
        return PlainTextResponse(as_text(payload), media_type="text/plain; charset=utf-8")

    if response_format == "json":
        return JSONResponse(as_json(payload))

    if response_format == "diarized_json":
        return JSONResponse(as_verbose_json(payload))

    if response_format == "verbose_json":
        return JSONResponse(as_verbose_json(payload))

    if response_format == "srt":
        return PlainTextResponse(as_srt(payload), media_type="text/plain; charset=utf-8")

    if response_format == "vtt":
        return PlainTextResponse(as_vtt(payload), media_type="text/plain; charset=utf-8")

    raise HTTPException(status_code=500, detail="Unexpected response format")
