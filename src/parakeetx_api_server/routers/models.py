from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import require_api_key
from ..deps import (
    get_diarization_manager,
    get_forced_alignment_manager,
    get_parakeet_manager,
    get_settings,
    get_vad_manager,
)
from ..model_managers.diarization_manager import DiarizationModelManager
from ..model_managers.forced_alignment_manager import ForcedAlignmentModelManager
from ..model_managers.parakeet_manager import ParakeetModelManager
from ..model_managers.vad_manager import VadModelManager

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("", dependencies=[Depends(require_api_key)])
async def list_models(
    parakeet: ParakeetModelManager = Depends(get_parakeet_manager),
):
    return {
        "object": "list",
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "created": 1715788800,
                "owned_by": "openai",
            },
            {
                "id": parakeet.configured_model_name,
                "object": "model",
                "created": 1715788800,
                "owned_by": "nvidia",
            },
        ],
    }


@router.get("/status", dependencies=[Depends(require_api_key)])
async def model_status(
    parakeet: ParakeetModelManager = Depends(get_parakeet_manager),
    diarization: DiarizationModelManager = Depends(get_diarization_manager),
    vad: VadModelManager = Depends(get_vad_manager),
    forced_alignment: ForcedAlignmentModelManager = Depends(get_forced_alignment_manager),
):
    return {
        "parakeet": parakeet.status(),
        "diarization": diarization.status(),
        "vad": vad.status(),
        "forced_alignment": forced_alignment.status(),
    }


@router.post("/parakeet/load", dependencies=[Depends(require_api_key)])
async def load_parakeet(
    parakeet: ParakeetModelManager = Depends(get_parakeet_manager),
):
    try:
        return parakeet.load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/parakeet/unload", dependencies=[Depends(require_api_key)])
async def unload_parakeet(
    parakeet: ParakeetModelManager = Depends(get_parakeet_manager),
):
    return parakeet.unload_model()


@router.post("/diarization/load", dependencies=[Depends(require_api_key)])
async def load_diarization(
    diarization: DiarizationModelManager = Depends(get_diarization_manager),
):
    try:
        return diarization.load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/diarization/unload", dependencies=[Depends(require_api_key)])
async def unload_diarization(
    diarization: DiarizationModelManager = Depends(get_diarization_manager),
):
    return diarization.unload_model()


@router.post("/vad/load", dependencies=[Depends(require_api_key)])
async def load_vad(
    vad: VadModelManager = Depends(get_vad_manager),
):
    try:
        return vad.load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/vad/unload", dependencies=[Depends(require_api_key)])
async def unload_vad(
    vad: VadModelManager = Depends(get_vad_manager),
):
    return vad.unload_model()


@router.post("/forced-alignment/load", dependencies=[Depends(require_api_key)])
async def load_forced_alignment(
    forced_alignment: ForcedAlignmentModelManager = Depends(get_forced_alignment_manager),
):
    if forced_alignment.settings.method != "qwen":
        return forced_alignment.status()
    try:
        return forced_alignment.load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/forced-alignment/unload", dependencies=[Depends(require_api_key)])
async def unload_forced_alignment(
    forced_alignment: ForcedAlignmentModelManager = Depends(get_forced_alignment_manager),
):
    return forced_alignment.unload_model()
