"""Domain exception hierarchy shared across the API and worker.

Every exception carries a stable `code` (used in API error responses,
PRODUCT_SPEC.md-style) and a `status_code` (the HTTP status the API layer
maps it to — see app/main.py's exception handlers). Worker code catches
these same exceptions to decide retry behavior and to build a safe
Episode.last_error / ProcessingJob.error_message string.
"""


class AppError(Exception):
    """Base class for all domain errors. Do not raise directly."""

    code: str = "INTERNAL_ERROR"
    status_code: int = 500

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
