from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, loaded from environment variables.

    See .env.example at the repo root for the full, documented list.
    Nothing here should ever hold a literal secret default.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/podcast_to_knowledge"
    redis_url: str = "redis://localhost:6379/0"

    supadata_api_key: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    embedding_model: str = ""
    vector_dimension: int = 1536

    chunk_target_tokens: int = 800
    chunk_min_tokens: int = 300
    chunk_max_tokens: int = 1200
    chunk_overlap_tokens: int = 100

    max_revision_attempts: int = 2

    admin_auth_secret: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
