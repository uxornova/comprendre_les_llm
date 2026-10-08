"""Middleware : mesure le temps de traitement de chaque requête et l'ajoute dans un en-tête de la réponse."""
import time

from starlette.middleware.base import BaseHTTPMiddleware


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.0f}"
        return response
