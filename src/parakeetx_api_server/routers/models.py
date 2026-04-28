from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import require_api_key
from ..deps import get_diarization_manager, get_parakeet_manager
from ..model_managers.diarization_manager import DiarizationModelManager
from ..model_managers.parakeet_manager import ParakeetModelManager

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("/status", dependencies=[Depends(require_api_key)])
async def model_status(
    parakeet: ParakeetModelManager = Depends(get_parakeet_manager),
    diarization: DiarizationModelManager = Depends(get_diarization_manager),
):
    return {
        "parakeet": parakeet.status(),
        "diarization": diarization.status(),
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
