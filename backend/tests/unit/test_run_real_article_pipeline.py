"""Tests for scripts/run_real_article_pipeline.py -- all HTTP calls are
mocked with respx, so this makes no real network call and, critically,
never calls a real LLM provider: the whole point of these tests is to
pin down that the script never even *requests* article generation unless
--generate-article is explicitly passed (the safety behavior the script
exists for).
"""

import json

import httpx
import pytest
import respx

from scripts.run_real_article_pipeline import (
    create_or_reuse_episode,
    fetch_article,
    format_article_report,
    run,
    trigger_article_generation,
    wait_for_status,
)

API_BASE_URL = "http://test-api"
VIDEO_URL = "https://www.youtube.com/watch?v=Y566_T-YlNQ"
EPISODE_ID = "11111111-1111-1111-1111-111111111111"


def _episode(status: str, last_error: str | None = None) -> dict:
    return {
        "id": EPISODE_ID,
        "youtube_video_id": "Y566_T-YlNQ",
        "status": status,
        "title": "Real Episode",
        "last_error": last_error,
    }


def _article() -> dict:
    return {
        "id": "22222222-2222-2222-2222-222222222222",
        "episode_id": EPISODE_ID,
        "title": "The Generated Article",
        "revision_count": 1,
        "episode_status": "READY_FOR_REVIEW",
        "sections": [
            {
                "sequence_number": 0,
                "heading": "Introduction",
                "content": "Generated prose. " * 20,
                "supporting_chunks": [
                    {"id": "c1", "start_ms": 1_000, "end_ms": 65_000, "text_preview": "source text"}
                ],
            }
        ],
        "latest_validation": {
            "passed": True,
            "checks": [{"name": "source_coverage", "passed": True, "details": "1/1 chunks referenced"}],
        },
    }


class _Args:
    def __init__(self, **kwargs):
        self.youtube_url = VIDEO_URL
        self.api_base_url = API_BASE_URL
        self.episode_id = None
        self.generate_article = False
        self.poll_interval = 0.001
        self.timeout = 5.0
        self.full_text = False
        self.__dict__.update(kwargs)


# --- individual HTTP helpers ---------------------------------------------------------


@respx.mock
async def test_create_or_reuse_episode_posts_the_url() -> None:
    route = respx.post(f"{API_BASE_URL}/api/v1/episodes").mock(
        return_value=httpx.Response(201, json=_episode("INGESTING"))
    )
    async with httpx.AsyncClient() as client:
        episode = await create_or_reuse_episode(client, API_BASE_URL, VIDEO_URL)

    assert route.called
    assert json.loads(route.calls.last.request.content) == {"youtube_url": VIDEO_URL}
    assert episode["id"] == EPISODE_ID


@respx.mock
async def test_wait_for_status_stops_once_a_ready_status_is_reached() -> None:
    respx.get(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}").mock(
        side_effect=[
            httpx.Response(200, json=_episode("INGESTING")),
            httpx.Response(200, json=_episode("TRANSCRIPT_FETCHED")),
            httpx.Response(200, json=_episode("CHUNKING")),
        ]
    )
    async with httpx.AsyncClient() as client:
        episode = await wait_for_status(
            client, API_BASE_URL, EPISODE_ID,
            ready_statuses={"CHUNKING"}, poll_interval=0.001, timeout=5.0, label="test",
        )

    assert episode["status"] == "CHUNKING"


@respx.mock
async def test_wait_for_status_stops_on_failed_without_raising() -> None:
    respx.get(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}").mock(
        return_value=httpx.Response(200, json=_episode("FAILED", last_error="Supadata error"))
    )
    async with httpx.AsyncClient() as client:
        episode = await wait_for_status(
            client, API_BASE_URL, EPISODE_ID,
            ready_statuses={"CHUNKING"}, poll_interval=0.001, timeout=5.0, label="test",
        )

    assert episode["status"] == "FAILED"
    assert episode["last_error"] == "Supadata error"


@respx.mock
async def test_wait_for_status_times_out_if_never_ready() -> None:
    respx.get(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}").mock(
        return_value=httpx.Response(200, json=_episode("INGESTING"))
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(TimeoutError):
            await wait_for_status(
                client, API_BASE_URL, EPISODE_ID,
                ready_statuses={"CHUNKING"}, poll_interval=0.001, timeout=0.001, label="test",
            )


@respx.mock
async def test_trigger_article_generation_posts_the_endpoint() -> None:
    route = respx.post(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}/generate-article").mock(
        return_value=httpx.Response(202, json={"episode_id": EPISODE_ID, "job_id": "job-1"})
    )
    async with httpx.AsyncClient() as client:
        result = await trigger_article_generation(client, API_BASE_URL, EPISODE_ID)

    assert route.called
    assert result["job_id"] == "job-1"


@respx.mock
async def test_fetch_article_gets_the_endpoint() -> None:
    respx.get(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}/article").mock(
        return_value=httpx.Response(200, json=_article())
    )
    async with httpx.AsyncClient() as client:
        article = await fetch_article(client, API_BASE_URL, EPISODE_ID)

    assert article["title"] == "The Generated Article"


# --- format_article_report: pure formatting -------------------------------------------


def test_format_article_report_includes_title_sections_and_validation() -> None:
    report = format_article_report(_article())

    assert "The Generated Article" in report
    assert "revision_count=1" in report
    assert "PASSED" in report
    assert "source_coverage" in report
    assert "Introduction" in report
    assert "00:01-01:05" in report  # 1_000ms -> 65_000ms formatted


def test_format_article_report_truncates_long_sections_by_default() -> None:
    article = _article()
    article["sections"][0]["content"] = "x" * 1000
    report = format_article_report(article, preview_chars=50)
    assert "..." in report


def test_format_article_report_full_text_skips_truncation() -> None:
    article = _article()
    article["sections"][0]["content"] = "x" * 1000
    report = format_article_report(article, full_text=True, preview_chars=50)
    assert "x" * 1000 in report


def test_format_article_report_handles_no_validation_result() -> None:
    article = _article()
    article["latest_validation"] = None
    report = format_article_report(article)
    assert "no result recorded" in report


# --- run(): the safety behavior the script exists for ----------------------------------


@respx.mock
async def test_run_stops_before_generate_article_by_default(capsys) -> None:
    respx.post(f"{API_BASE_URL}/api/v1/episodes").mock(
        return_value=httpx.Response(201, json=_episode("INGESTING"))
    )
    respx.get(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}").mock(
        return_value=httpx.Response(200, json=_episode("CHUNKING"))
    )
    generate_route = respx.post(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}/generate-article")
    article_route = respx.get(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}/article")

    await run(_Args(generate_article=False))

    assert not generate_route.called
    assert not article_route.called
    output = capsys.readouterr().out
    assert "no LLM calls made" in output
    assert "--generate-article" in output


@respx.mock
async def test_run_generates_the_article_only_when_flag_is_passed(capsys) -> None:
    respx.post(f"{API_BASE_URL}/api/v1/episodes").mock(
        return_value=httpx.Response(201, json=_episode("INGESTING"))
    )
    respx.get(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}").mock(
        side_effect=[
            httpx.Response(200, json=_episode("CHUNKING")),
            httpx.Response(200, json=_episode("READY_FOR_REVIEW")),
        ]
    )
    generate_route = respx.post(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}/generate-article").mock(
        return_value=httpx.Response(202, json={"episode_id": EPISODE_ID, "job_id": "job-1"})
    )
    respx.get(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}/article").mock(
        return_value=httpx.Response(200, json=_article())
    )

    await run(_Args(generate_article=True))

    assert generate_route.called
    output = capsys.readouterr().out
    assert "The Generated Article" in output
    assert "PASSED" in output


@respx.mock
async def test_run_stops_on_ingestion_failure_without_calling_generate(capsys) -> None:
    respx.post(f"{API_BASE_URL}/api/v1/episodes").mock(
        return_value=httpx.Response(201, json=_episode("INGESTING"))
    )
    respx.get(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}").mock(
        return_value=httpx.Response(200, json=_episode("FAILED", last_error="bad video"))
    )
    generate_route = respx.post(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}/generate-article")

    with pytest.raises(SystemExit):
        await run(_Args(generate_article=True))

    assert not generate_route.called


@respx.mock
async def test_run_reuses_an_existing_episode_id_without_posting_episodes() -> None:
    create_route = respx.post(f"{API_BASE_URL}/api/v1/episodes")
    respx.get(f"{API_BASE_URL}/api/v1/episodes/{EPISODE_ID}").mock(
        return_value=httpx.Response(200, json=_episode("CHUNKING"))
    )

    await run(_Args(episode_id=EPISODE_ID, generate_article=False))

    assert not create_route.called
