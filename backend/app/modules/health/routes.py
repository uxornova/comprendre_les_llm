"""Route de santé : utilisée par le front au chargement et par le healthcheck Docker."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.modules.llm.service import llm_service

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health():
    if not llm_service.ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", **llm_service.info()}
