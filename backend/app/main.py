"""Point d'entrée du back (équivalent de server.ts dans Matcha).

Développement (depuis la racine du projet) :  make dev
Production : docker compose up --build
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.middleware.timing import TimingMiddleware
from app.modules.health.routes import router as health_router
from app.modules.llm.routes import router as llm_router
from app.modules.llm.service import llm_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm_service.load()
    print(f"✅ {settings.model_name} chargé en {llm_service.load_seconds:.1f} s")
    yield


app = FastAPI(title="Comprendre les LLM — API", lifespan=lifespan)
app.add_middleware(TimingMiddleware)
app.include_router(health_router)
app.include_router(llm_router)

# En développement, le back sert aussi les fichiers du front (en production : nginx)
if settings.frontend_dir:
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
