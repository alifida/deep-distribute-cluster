"""
Entrypoint script for deep-distribute.

Bootstraps the application based on role (ps | worker), loads configuration,
and starts FastAPI server.
"""

import sys
import uvicorn
from fastapi import FastAPI

from shared.config import settings


# Routers (to be implemented in ps.py / worker.py)
# For now placeholders to ensure boot works.
try:
    if settings.ROLE == "ps":
        from ps import router as ps_router
    elif settings.ROLE == "worker":
        from worker import router as worker_router
    else:
        raise ValueError(f"Invalid ROLE: {settings.ROLE}")
except ImportError:
    ps_router = None
    worker_router = None


# ----------------------------
# App factory
# ----------------------------
def create_app() -> FastAPI:
    app = FastAPI(title="Deep Distribute", version="0.1.0")

    if settings.ROLE == "ps" and ps_router:
        app.include_router(ps_router, prefix="/ps", tags=["ParameterServer"])
    elif settings.ROLE == "worker" and worker_router:
        app.include_router(worker_router, prefix="/worker", tags=["Worker"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "role": settings.ROLE}

    return app


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    app = create_app()
    uvicorn.run(
        "entrypoint:create_app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        factory=True,
    )
