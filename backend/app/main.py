from fastapi import FastAPI

from app.api.v1 import api_router
from app.config import get_settings
from app.core.error_handlers import register_error_handlers


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Podcast-to-Knowledge Platform API",
        version="0.1.0",
        debug=settings.app_env == "development",
    )

    register_error_handlers(app)
    app.include_router(api_router)

    return app


app = create_app()
