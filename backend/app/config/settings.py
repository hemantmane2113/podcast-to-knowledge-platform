from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Secrets that must be configured in any environment other than local
# development. Development stays runnable without them since nothing in
# the dev inner loop should require real provider credentials just to boot
# the app; anything that actually needs one of these will fail clearly at
# the point of use (e.g. the transcript provider) instead.
_REQUIRED_OUTSIDE_DEVELOPMENT = ("supadata_api_key", "llm_api_key", "admin_auth_secret")


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

    @model_validator(mode="after")
    def _require_secrets_outside_development(self) -> "Settings":
        if self.app_env == "development":
            return self

        missing = [name for name in _REQUIRED_OUTSIDE_DEVELOPMENT if not getattr(self, name)]
        if missing:
            env_var_names = ", ".join(name.upper() for name in missing)
            raise ValueError(
                f"APP_ENV={self.app_env!r} requires the following environment "
                f"variables to be set: {env_var_names}"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
