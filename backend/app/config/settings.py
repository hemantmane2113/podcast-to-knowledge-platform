from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Secrets that must be configured in any environment other than local
# development, regardless of which LLM provider is selected. Development
# stays runnable without them since nothing in the dev inner loop should
# require real provider credentials just to boot the app; anything that
# actually needs one of these will fail clearly at the point of use (e.g.
# the transcript provider) instead. The LLM provider's own key is checked
# separately below, since which key is required depends on LLM_PROVIDER —
# "only the selected provider should require its API key" (see
# app/providers/llm/factory.py).
_REQUIRED_OUTSIDE_DEVELOPMENT = ("supadata_api_key", "admin_auth_secret")

_LLM_PROVIDER_KEY_FIELD = {
    "groq": "groq_api_key",
    "openai": "openai_api_key",
    # Self-hosted/OpenAI-compatible endpoints (vLLM, Ollama, TGI, ...)
    # don't require a real key -- only a reachable base_url -- so this is
    # deliberately not in the outside-development requirement below.
    "opensource": "opensource_base_url",
}


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

    # --- LLM provider (Phase C+: topic analysis, planning, generation,
    # validation) -- app/providers/llm/. Application code depends on the
    # LLMProvider abstraction, never on a provider SDK directly; this is
    # the one place a provider is selected.
    llm_provider: str = "groq"  # groq | openai | opensource
    llm_model: str = ""
    groq_api_key: str = ""
    openai_api_key: str = ""
    opensource_base_url: str = ""
    opensource_api_key: str = ""
    # Off by default: an LLM-based validation check is a supplementary
    # signal, never a replacement for the deterministic checks in
    # app/services/article_validation.py, and shouldn't add a paid call to
    # every run until explicitly opted into.
    enable_llm_validation: bool = False

    embedding_model: str = ""
    vector_dimension: int = 1536

    # Target size a chunk tries to reach before looking for a good place to
    # stop (sentence end / pause gap). Not a hard limit.
    chunk_target_tokens: int = 800
    # Below this, a chunk boundary is skipped in favor of continuing to
    # accumulate more segments (except the transcript's trailing chunk,
    # which may legitimately be smaller).
    chunk_min_tokens: int = 300
    # Hard ceiling: a chunk is always cut at or before this, even if that
    # means splitting a single long segment's text (see chunking_service.py).
    chunk_max_tokens: int = 1200
    # Segment-granular overlap between adjacent chunks, in approximate
    # tokens. Defaults to 0 (disabled): nothing in Phase 3A consumes
    # overlap (no embeddings/RAG yet — see ARCHITECTURE.md), so shipping
    # it non-zero by default would just duplicate text with no current
    # benefit. The mechanism is fully implemented and tested; set this
    # above 0 once a real consumer (Phase 3C RAG) needs it.
    chunk_overlap_tokens: int = 0
    # A gap this long (or longer) between two consecutive segments is
    # treated as a candidate chunk boundary (a natural pause in speech),
    # one signal among several — never an absolute cut rule on its own.
    chunk_pause_threshold_ms: int = 1500

    max_revision_attempts: int = 2

    # Rough token budget for one topic-analysis LLM call's transcript text
    # (chunks' combined text, not the whole prompt) -- a long-form podcast's
    # canonical chunks can together exceed most models' usable context, so
    # topic analysis batches chunks under this budget per call and merges
    # the per-batch results (app/ai/nodes/topic_analysis.py). Conservative
    # default well under typical 32k-128k context windows, leaving room for
    # the prompt scaffolding and the model's own output.
    topic_analysis_token_budget: int = 12000
    # Soft ceiling on generated-article length as a fraction of the source
    # transcript's word count -- the product goal is "substantially shorter
    # than the original conversation" (see PRODUCT_SPEC.md), not a hard
    # limit: article_validation.py reports a violation, it doesn't block.
    article_max_length_ratio: float = 0.4

    admin_auth_secret: str = ""

    @model_validator(mode="after")
    def _require_secrets_outside_development(self) -> "Settings":
        if self.app_env == "development":
            return self

        missing = [name for name in _REQUIRED_OUTSIDE_DEVELOPMENT if not getattr(self, name)]

        key_field = _LLM_PROVIDER_KEY_FIELD.get(self.llm_provider)
        if key_field is None:
            raise ValueError(
                f"LLM_PROVIDER={self.llm_provider!r} is not one of "
                f"{sorted(_LLM_PROVIDER_KEY_FIELD)}"
            )
        if not getattr(self, key_field):
            missing.append(key_field)

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
