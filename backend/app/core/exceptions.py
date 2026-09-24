"""Domain exception hierarchy shared across the API and worker.

Every exception carries a stable `code` (used in API error responses,
PRODUCT_SPEC.md-style) and a `status_code` (the HTTP status the API layer
maps it to — see app/main.py's exception handlers). Worker code catches
these same exceptions to decide retry behavior and to build a safe
Episode.last_error / ProcessingJob.error_message string.
"""


class AppError(Exception):
    """Base class for all domain errors. Do not raise directly.

    `retryable` defaults to False deliberately: most domain errors (not
    found, already processing, validation) describe state that won't
    change on retry. Only TranscriptProviderError -- genuinely transient
    external-call failures -- flips the default back to True, and its
    subclasses override it individually where retrying is actually
    pointless (auth, not-found, malformed response). Found via a real bug:
    TranscriptNotFoundError (this module) had no `retryable` at all before
    this, so worker code's `getattr(exc, "retryable", True)` silently
    treated "no transcript exists for this episode" as worth retrying,
    which can never succeed.
    """

    code: str = "INTERNAL_ERROR"
    status_code: int = 500
    retryable: bool = False

    def __init__(self, message: str | None = None):
        self.message = message or self.code
        super().__init__(self.message)


class EpisodeNotFoundError(AppError):
    code = "EPISODE_NOT_FOUND"
    status_code = 404


class TranscriptNotFoundError(AppError):
    """The episode exists but has no transcript yet (still ingesting, or
    ingestion failed before a transcript was persisted)."""

    code = "TRANSCRIPT_NOT_FOUND"
    status_code = 404


class EpisodeAlreadyProcessingError(AppError):
    """A processing job for this episode is already running."""

    code = "EPISODE_ALREADY_PROCESSING"
    status_code = 409


class ChunksNotFoundError(AppError):
    """The episode has a transcript but no chunks yet -- Phase 3A
    (cleaning + chunking) hasn't completed for it, so article generation
    has nothing to work from."""

    code = "CHUNKS_NOT_FOUND"
    status_code = 404


class ArticleNotFoundError(AppError):
    """No article has been generated for this episode yet -- also raised
    (deliberately) by the public read path for an article that exists
    but isn't PUBLISHED, so an unpublished article's existence is never
    distinguishable from "no article at all" to a public caller."""

    code = "ARTICLE_NOT_FOUND"
    status_code = 404


class ArticleValidationNotPassedError(AppError):
    """The article's latest ValidationResult doesn't exist or didn't
    pass (app/services/article_validation.py) -- publishing requires a
    validated article (generate -> validate -> human review -> explicit
    publish); this is never bypassed."""

    code = "ARTICLE_VALIDATION_NOT_PASSED"
    status_code = 409


class TranscriptProviderError(AppError):
    """The transcript provider (e.g. Supadata) failed in a way that isn't
    one of the more specific subclasses below. Used as the catch-all
    provider error and as the base class the worker retries on by default.
    """

    code = "TRANSCRIPT_PROVIDER_ERROR"
    status_code = 502
    retryable = True


class TranscriptProviderAuthError(TranscriptProviderError):
    """Invalid/missing provider API key. Never retried — retrying a bad
    credential just wastes attempts and quota."""

    retryable = False


class TranscriptProviderRateLimitError(TranscriptProviderError):
    """Provider rate limit hit. Retried with backoff."""

    retryable = True


class TranscriptProviderNotFoundError(TranscriptProviderError):
    """The provider has no transcript for this video (captions disabled,
    video doesn't exist, etc.). Never retried — the outcome won't change."""

    retryable = False


class MalformedProviderResponseError(TranscriptProviderError):
    """The provider returned a response that doesn't match the documented
    shape. Never retried — a malformed response won't fix itself, and
    silently guessing at fields risks inventing transcript content."""

    retryable = False


class LLMProviderError(AppError):
    """An LLMProvider (Groq/OpenAI/open-source) call failed in a way that
    isn't one of the more specific subclasses below. Mirrors
    TranscriptProviderError's shape and defaults (see app/providers/llm/)."""

    code = "LLM_PROVIDER_ERROR"
    status_code = 502
    retryable = True


class LLMProviderAuthError(LLMProviderError):
    """Invalid/missing provider API key. Never retried."""

    code = "LLM_PROVIDER_AUTH_ERROR"
    retryable = False


class LLMProviderRateLimitError(LLMProviderError):
    """Provider rate limit hit. Retried with backoff."""

    code = "LLM_PROVIDER_RATE_LIMIT_ERROR"
    retryable = True


class LLMProviderTransientError(LLMProviderError):
    """A server-side (5xx) or connection/timeout failure -- distinct from
    the rate-limit and generic-unknown cases mainly so logging/metrics can
    tell them apart; behaves identically to the LLMProviderError default
    otherwise (retryable)."""

    code = "LLM_PROVIDER_TRANSIENT_ERROR"
    retryable = True


class LLMProviderRequestError(LLMProviderError):
    """The request itself was malformed (HTTP 400) -- an application-side
    bug (bad params, invalid schema, ...), not a provider outage. Never
    retried: retrying the same bad request against a different provider
    wouldn't fix it either, and would hide the real bug."""

    code = "LLM_PROVIDER_REQUEST_ERROR"
    retryable = False


class LLMStructuredOutputError(LLMProviderError):
    """The model's response couldn't be parsed/validated against the
    requested structured-output schema, even after one retry. Never
    retried further here — a third identical attempt is unlikely to
    differ; the caller decides whether to fall back or fail the job."""

    code = "LLM_STRUCTURED_OUTPUT_ERROR"
    retryable = False
