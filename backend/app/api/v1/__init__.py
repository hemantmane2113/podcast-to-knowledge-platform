from fastapi import APIRouter

from app.api.v1.articles import router as articles_router
from app.api.v1.episodes import router as episodes_router
from app.api.v1.health import router as health_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health_router)
api_router.include_router(episodes_router)
api_router.include_router(articles_router)
