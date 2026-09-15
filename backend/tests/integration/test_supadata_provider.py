from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from app.core.exceptions import (
    MalformedProviderResponseError,
    TranscriptProviderAuthError,
    TranscriptProviderError,
    TranscriptProviderNotFoundError,
    TranscriptProviderRateLimitError,
)
from app.providers.transcript.supadata import SupadataTranscriptProvider

BASE_URL = "https://api.supadata.test/v1"
VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def provider() -> SupadataTranscriptProvider:
    return SupadataTranscriptProvider(api_key="test-key", base_url=BASE_URL)


def test_rejects_missing_api_key() -> None:
    with pytest.raises(TranscriptProviderAuthError):
        SupadataTranscriptProvider(api_key="")


@respx.mock
async def test_get_transcript_success(provider: SupadataTranscriptProvider) -> None:
    route = respx.get(f"{BASE_URL}/transcript").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [
                    {"text": "Never gonna give you up", "offset": 18400, "duration": 2040, "lang": "en"},
                    {"text": "Never gonna let you down", "offset": 20440, "duration": 2080, "lang": "en"},
                ],
                "lang": "en",
                "availableLangs": ["en"],
            },
        )
    )

    transcript = await provider.get_transcript(VIDEO_URL)

    assert route.called
    request = route.calls.last.request
    assert request.headers["x-api-key"] == "test-key"
    assert request.url.params["url"] == VIDEO_URL
    assert request.url.params["mode"] == "native"

    assert transcript.language == "en"
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "Never gonna give you up"
    assert transcript.segments[0].start_ms == 18400
    assert transcript.segments[0].duration_ms == 2040
    assert transcript.segments[0].speaker is None


@respx.mock
async def test_get_transcript_async_job_polling(provider: SupadataTranscriptProvider) -> None:
    respx.get(f"{BASE_URL}/transcript").mock(return_value=httpx.Response(202, json={"jobId": "job-123"}))
    respx.get(f"{BASE_URL}/transcript/job-123").mock(
        side_effect=[
            httpx.Response(200, json={"status": "queued"}),
            httpx.Response(200, json={"status": "active"}),
            httpx.Response(
                200,
                json={
                    "status": "completed",
                    "content": [{"text": "hello", "offset": 0, "duration": 500, "lang": "en"}],
                    "lang": "en",
                },
            ),
        ]
    )

    with patch("app.providers.transcript.supadata.asyncio.sleep", new=AsyncMock()):
        transcript = await provider.get_transcript(VIDEO_URL)

    assert transcript.segments[0].text == "hello"


@respx.mock
async def test_get_transcript_not_found(provider: SupadataTranscriptProvider) -> None:
    respx.get(f"{BASE_URL}/transcript").mock(
        return_value=httpx.Response(
            404,
            json={
                "error": "transcript-unavailable",
                "message": "Transcript Unavailable",
                "details": "No transcript is available for this video",
            },
        )
    )

    with pytest.raises(TranscriptProviderNotFoundError):
        await provider.get_transcript(VIDEO_URL)


@respx.mock
async def test_get_transcript_empty_content_treated_as_not_found(
    provider: SupadataTranscriptProvider,
) -> None:
    respx.get(f"{BASE_URL}/transcript").mock(
        return_value=httpx.Response(200, json={"content": [], "lang": "en"})
    )

    with pytest.raises(TranscriptProviderNotFoundError):
        await provider.get_transcript(VIDEO_URL)


@respx.mock
async def test_get_transcript_auth_failure(provider: SupadataTranscriptProvider) -> None:
    respx.get(f"{BASE_URL}/transcript").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized", "message": "Invalid API key"})
    )

    with pytest.raises(TranscriptProviderAuthError):
        await provider.get_transcript(VIDEO_URL)


@respx.mock
async def test_get_transcript_rate_limit(provider: SupadataTranscriptProvider) -> None:
    respx.get(f"{BASE_URL}/transcript").mock(
        return_value=httpx.Response(429, json={"error": "rate-limited", "message": "Too many requests"})
    )

    with pytest.raises(TranscriptProviderRateLimitError):
        await provider.get_transcript(VIDEO_URL)


@respx.mock
async def test_get_transcript_generic_provider_error(provider: SupadataTranscriptProvider) -> None:
    respx.get(f"{BASE_URL}/transcript").mock(
        return_value=httpx.Response(500, json={"error": "internal", "message": "Something broke"})
    )

    with pytest.raises(TranscriptProviderError):
        await provider.get_transcript(VIDEO_URL)


@respx.mock
async def test_get_transcript_malformed_response_missing_content(
    provider: SupadataTranscriptProvider,
) -> None:
    respx.get(f"{BASE_URL}/transcript").mock(return_value=httpx.Response(200, json={"lang": "en"}))

    with pytest.raises(MalformedProviderResponseError):
        await provider.get_transcript(VIDEO_URL)


@respx.mock
async def test_get_transcript_malformed_response_bad_segment_shape(
    provider: SupadataTranscriptProvider,
) -> None:
    respx.get(f"{BASE_URL}/transcript").mock(
        return_value=httpx.Response(200, json={"content": [{"text": "missing offset/duration"}]})
    )

    with pytest.raises(MalformedProviderResponseError):
        await provider.get_transcript(VIDEO_URL)


@respx.mock
async def test_get_transcript_non_json_response(provider: SupadataTranscriptProvider) -> None:
    respx.get(f"{BASE_URL}/transcript").mock(
        return_value=httpx.Response(200, content=b"<html>not json</html>")
    )

    with pytest.raises(MalformedProviderResponseError):
        await provider.get_transcript(VIDEO_URL)


@respx.mock
async def test_get_metadata_success(provider: SupadataTranscriptProvider) -> None:
    respx.get(f"{BASE_URL}/metadata").mock(
        return_value=httpx.Response(
            200,
            json={
                "title": "Never Gonna Give You Up",
                "description": "The official video",
                "author": {"name": "Rick Astley", "id": "UC_channel"},
                "media": {"duration": 213, "thumbnailUrl": "https://example.com/thumb.jpg"},
                "createdAt": "2009-10-25T06:57:33Z",
                "additionalData": {"channelId": "UC_channel"},
            },
        )
    )

    metadata = await provider.get_metadata(VIDEO_URL)

    assert metadata.title == "Never Gonna Give You Up"
    assert metadata.channel_name == "Rick Astley"
    assert metadata.channel_id == "UC_channel"
    assert metadata.duration_seconds == 213
    assert metadata.thumbnail_url == "https://example.com/thumb.jpg"
    assert metadata.published_at == "2009-10-25T06:57:33Z"


@respx.mock
async def test_get_metadata_missing_fields_are_none_not_fabricated(
    provider: SupadataTranscriptProvider,
) -> None:
    respx.get(f"{BASE_URL}/metadata").mock(return_value=httpx.Response(200, json={"title": "Only a title"}))

    metadata = await provider.get_metadata(VIDEO_URL)

    assert metadata.title == "Only a title"
    assert metadata.channel_name is None
    assert metadata.duration_seconds is None
    assert metadata.published_at is None
