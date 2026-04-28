from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import require_api_key

router = APIRouter(prefix="/v1/audio", tags=["audio"])


@router.post("/translations", dependencies=[Depends(require_api_key)])
async def create_translation():
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="/v1/audio/translations is not implemented for ParakeetX.",
    )
