"""Supadata (https://supadata.ai/) transcript provider.

Endpoint shapes below are based on Supadata's public API reference
(docs.supadata.ai) as of the time this was written. Supadata's docs site
was not directly fetchable from this environment (network policy), so this
was assembled from third-party summaries of the documented behavior rather
than the canonical page — verify against a live request (or the current
docs) before pointing this at a production API key, per PRODUCT_SPEC.md §12
("do not assume the API response schema if documentation differs").

Assumed contract:
  GET {base}/transcript?url=<video_url>&mode=native&text=false
    -> 200 {"content": [{"text","offset","duration","lang"}, ...],
             "lang": "en", "availableLangs": ["en", ...]}
    -> 202 {"jobId": "..."}  (large videos process asynchronously;
             poll GET {base}/transcript/{jobId} until status is
             "completed" or "failed")
    -> 4xx/5xx {"error": "<code>", "message": "...", "details": "...",
                 "documentationUrl": "..."}
  GET {base}/metadata?url=<video_url>
    -> 200 {"title", "description", "author": {"name","id"},
             "media": {"duration","thumbnailUrl"}, "createdAt",
             "additionalData": {"channelId"}, ...}
  Auth: header "x-api-key: <SUPADATA_API_KEY>"

mode=native (rather than mode=auto) is used deliberately: it returns only
transcripts that already exist (real captions) instead of paying to
AI-generate one, which keeps the transcript closer to an authoritative
source per PRODUCT_SPEC.md §88 ("never invent"). A video with no existing
captions will surface as TranscriptProviderNotFoundError rather than
silently falling back to a generated transcript.
"""

import asyncio

import httpx

from app.core.exceptions import (
    MalformedProviderResponseError,
    TranscriptProviderAuthError,
    TranscriptProviderError,
    TranscriptProviderNotFoundError,
    TranscriptProviderRateLimitError,
)
from app.providers.transcript.base import TranscriptProvider
from app.schemas.transcript import EpisodeMetadata, NormalizedSegment, NormalizedTranscript

_DEFAULT_BASE_URL = "https://api.supadata.ai/v1"
_POLL_INTERVAL_SECONDS = 2.0
_POLL_MAX_ATTEMPTS = 30  # ~60s bound on async transcript-job polling


class SupadataTranscriptProvider(TranscriptProvider):
    def __init__(
        self,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        base_url: str = _DEFAULT_BASE_URL,
    ):
        if not api_key:
            raise TranscriptProviderAuthError("SUPADATA_API_KEY is not configured")
        self._api_key = api_key
        self._base_url = base_url
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_transcript(self, video_url: str) -> NormalizedTranscript:
        response = await self._request(
            "GET",
            "/transcript",
            params={"url": video_url, "mode": "native", "text": "false"},
        )
        payload = self._parse_json(response)

        if response.status_code == 202:
            job_id = payload.get("jobId")
            if not job_id:
                raise MalformedProviderResponseError("Supadata returned 202 without a jobId")
            payload = await self._poll_transcript_job(job_id)

        return self._normalize_transcript(payload)

    async def get_metadata(self, video_url: str) -> EpisodeMetadata:
        response = await self._request("GET", "/metadata", params={"url": video_url})
        payload = self._parse_json(response)
        return self._normalize_metadata(payload)

    async def _poll_transcript_job(self, job_id: str) -> dict:
        for _ in range(_POLL_MAX_ATTEMPTS):
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            response = await self._request("GET", f"/transcript/{job_id}")
            payload = self._parse_json(response)
            status = payload.get("status")
            if status == "completed":
                return payload
            if status == "failed":
                raise TranscriptProviderError(
                    payload.get("message") or f"Supadata transcript job {job_id} failed"
                )
            # "queued" / "active" -> keep polling
        raise TranscriptProviderError(f"Timed out waiting for Supadata transcript job {job_id}")

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = await self._client.request(
                method,
                f"{self._base_url}{path}",
                headers={"x-api-key": self._api_key},
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise TranscriptProviderError("Supadata request timed out") from exc
        except httpx.HTTPError as exc:
            raise TranscriptProviderError(
                f"Supadata request failed: {exc.__class__.__name__}"
            ) from exc

        if response.status_code not in (200, 202):
            self._raise_for_error(response)

        return response

    def _raise_for_error(self, response: httpx.Response) -> None:
        payload = self._parse_json(response, allow_empty=True)
        error_code = payload.get("error")
        message = payload.get("message") or error_code or "Supadata request failed"

        if response.status_code == 401:
            raise TranscriptProviderAuthError(message)
        if response.status_code == 429:
            raise TranscriptProviderRateLimitError(message)
        if response.status_code == 404 or error_code == "transcript-unavailable":
            raise TranscriptProviderNotFoundError(message)
        raise TranscriptProviderError(f"Supadata returned {response.status_code}: {message}")

    def _parse_json(self, response: httpx.Response, allow_empty: bool = False) -> dict:
        try:
            return response.json()
        except ValueError as exc:
            if allow_empty:
                return {}
            raise MalformedProviderResponseError(
                f"Supadata returned a non-JSON response (status {response.status_code})"
            ) from exc

    def _normalize_transcript(self, payload: dict) -> NormalizedTranscript:
        content = payload.get("content")
        if not isinstance(content, list):
            raise MalformedProviderResponseError("Supadata response is missing a 'content' array")

        segments: list[NormalizedSegment] = []
        for raw_segment in content:
            try:
                segments.append(
                    NormalizedSegment(
                        text=raw_segment["text"],
                        start_ms=raw_segment["offset"],
                        duration_ms=raw_segment["duration"],
                        # The transcript endpoint doesn't return diarization
                        # -- never invent a speaker label.
                        speaker=None,
                    )
                )
            except (KeyError, TypeError) as exc:
                raise MalformedProviderResponseError(
                    f"Supadata transcript segment missing expected fields: {exc}"
                ) from exc

        if not segments:
            raise TranscriptProviderNotFoundError("Supadata returned an empty transcript")

        return NormalizedTranscript(language=payload.get("lang"), segments=segments)

    def _normalize_metadata(self, payload: dict) -> EpisodeMetadata:
        # Best-effort and defensive: metadata is supplementary, so a field
        # this shape doesn't anticipate becomes None rather than an error
        # (unlike transcript content, which is load-bearing and fails loudly).
        media = payload.get("media") or {}
        author = payload.get("author") or {}
        additional_data = payload.get("additionalData") or {}

        return EpisodeMetadata(
            title=payload.get("title"),
            description=payload.get("description"),
            channel_name=author.get("name"),
            channel_id=author.get("id") or additional_data.get("channelId"),
            thumbnail_url=media.get("thumbnailUrl"),
            duration_seconds=media.get("duration"),
            published_at=payload.get("createdAt"),
            language=payload.get("language"),
        )
