"""Drives the real end-to-end V1 pipeline against a REAL running backend
via its actual HTTP API:

    YouTube URL -> POST /episodes (ingestion, auto-chains into cleaning
    + chunking) -> [stops here by default, no LLM cost] ->
    POST /episodes/{id}/generate-article (topic analysis -> planning ->
    section generation -> validation -> bounded revision, via
    app/ai/graph.py, run inside the real worker) -> GET /episodes/{id}/article

This is deliberately an HTTP client over the real API, not a second
implementation of the pipeline: app/ai/graph.py, the worker task
(app/worker/tasks.py::generate_article), and every repository are
exercised exactly as they are for any other caller -- this script only
automates the POST/poll/POST/poll/GET sequence documented in README.md.
A DB-free reimplementation was deliberately not built instead: Topic/
ArticlePlan/Article persistence is core to this pipeline's design (the
plan and each stage are meant to be independently inspectable), not
incidental, so bypassing the database would mean re-deriving that
persistence logic a second time rather than exercising the real one --
see ARCHITECTURE.md §17.5.

Requires the real stack already running (docker compose up, or the
native equivalent from README.md) with real credentials configured
SERVER-SIDE in its .env (SUPADATA_API_KEY, GROQ_API_KEY, LLM_PROVIDER,
LLM_MODEL). This script itself never reads, needs, or prints any of
those keys -- it only talks to the API over HTTP.

Stops BEFORE the paid article-generation step by default, once ingestion
and chunking finish, and prints the exact follow-up command -- pass
--generate-article to actually enqueue it (a real LLM/Groq call).

Usage:
    # Step 1: ingest + wait for chunking. No LLM cost. Review the output.
    python -m scripts.run_real_article_pipeline "https://www.youtube.com/watch?v=Y566_T-YlNQ"

    # Step 2, only after reviewing step 1: actually generate the article
    # (this makes real, paid calls to your configured LLM provider).
    python -m scripts.run_real_article_pipeline "https://www.youtube.com/watch?v=Y566_T-YlNQ" --generate-article

    # Resume against an episode a previous run already ingested:
    python -m scripts.run_real_article_pipeline "https://www.youtube.com/watch?v=Y566_T-YlNQ" \\
        --episode-id <uuid> --generate-article
"""

import argparse
import asyncio
import sys
import time

import httpx

_READY_FOR_GENERATION_STATUSES = {
    "CHUNKING",
    "ANALYZING",
    "PLANNING",
    "GENERATING",
    "VERIFYING",
    "REVISING",
    "READY_FOR_REVIEW",
    "PUBLISHED",
}
_ARTICLE_READY_STATUSES = {"READY_FOR_REVIEW", "PUBLISHED"}
_TERMINAL_FAILURE_STATUS = "FAILED"

_DEFAULT_POLL_INTERVAL_SECONDS = 5.0
_DEFAULT_TIMEOUT_SECONDS = 1800.0
_PREVIEW_CHARS = 400


async def create_or_reuse_episode(
    client: httpx.AsyncClient, api_base_url: str, youtube_url: str
) -> dict:
    """POST /episodes is idempotent server-side (EpisodeService) -- safe
    to call even if this exact video was already ingested."""
    response = await client.post(f"{api_base_url}/api/v1/episodes", json={"youtube_url": youtube_url})
    response.raise_for_status()
    return response.json()


async def get_episode(client: httpx.AsyncClient, api_base_url: str, episode_id: str) -> dict:
    response = await client.get(f"{api_base_url}/api/v1/episodes/{episode_id}")
    response.raise_for_status()
    return response.json()


async def wait_for_status(
    client: httpx.AsyncClient,
    api_base_url: str,
    episode_id: str,
    *,
    ready_statuses: set[str],
    poll_interval: float,
    timeout: float,
    label: str,
) -> dict:
    """Polls GET /episodes/{id} until status is in `ready_statuses` or
    FAILED. Never raises on FAILED -- the caller decides what that means
    (it's a normal, expected outcome to check for, not this function's
    business); raises TimeoutError only if neither is reached in time.
    """
    deadline = time.monotonic() + timeout
    while True:
        episode = await get_episode(client, api_base_url, episode_id)
        status = episode["status"]
        print(f"  [{label}] status={status}")
        if status in ready_statuses or status == _TERMINAL_FAILURE_STATUS:
            return episode
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {label} (last status: {status})")
        await asyncio.sleep(poll_interval)


async def trigger_article_generation(client: httpx.AsyncClient, api_base_url: str, episode_id: str) -> dict:
    response = await client.post(f"{api_base_url}/api/v1/episodes/{episode_id}/generate-article")
    response.raise_for_status()
    return response.json()


async def fetch_article(client: httpx.AsyncClient, api_base_url: str, episode_id: str) -> dict:
    response = await client.get(f"{api_base_url}/api/v1/episodes/{episode_id}/article")
    response.raise_for_status()
    return response.json()


def _format_timestamp(ms: int) -> str:
    total_seconds = ms // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_article_report(article: dict, *, full_text: bool = False, preview_chars: int = _PREVIEW_CHARS) -> str:
    lines = [f"Article: {article['title']!r}  (revision_count={article['revision_count']})"]

    validation = article.get("latest_validation")
    if validation:
        lines.append(f"Validation: {'PASSED' if validation['passed'] else 'FAILED'}")
        for check in validation["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"  [{mark}] {check['name']}: {check['details']}")
    else:
        lines.append("Validation: no result recorded")

    lines.append(f"\n{len(article['sections'])} section(s)\n")
    for section in article["sections"]:
        lines.append(f"## {section['sequence_number'] + 1}. {section['heading']}")
        sources = ", ".join(
            f"{_format_timestamp(c['start_ms'])}-{_format_timestamp(c['end_ms'])}"
            for c in section["supporting_chunks"]
        )
        lines.append(f"Sources: {sources or '(none)'}\n")
        text = section["content"] if full_text else section["content"][:preview_chars]
        if not full_text and len(section["content"]) > preview_chars:
            text += "..."
        lines.append(text)
        lines.append("\n" + "-" * 70 + "\n")

    return "\n".join(lines)


async def run(args: argparse.Namespace) -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        if args.episode_id:
            episode_id = args.episode_id
            print(f"Using existing episode {episode_id}")
        else:
            print(f"Creating (or reusing) episode for {args.youtube_url}")
            episode = await create_or_reuse_episode(client, args.api_base_url, args.youtube_url)
            episode_id = episode["id"]
            print(f"Episode {episode_id}: status={episode['status']}")

        print("\nWaiting for ingestion + cleaning + chunking...")
        episode = await wait_for_status(
            client,
            args.api_base_url,
            episode_id,
            ready_statuses=_READY_FOR_GENERATION_STATUSES,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            label="ingest+chunk",
        )
        if episode["status"] == _TERMINAL_FAILURE_STATUS:
            print(f"\nIngestion/processing FAILED: {episode.get('last_error')}", file=sys.stderr)
            raise SystemExit(1)
        print(f"\nChunking complete. Episode {episode_id} is ready for article generation.")

        if not args.generate_article:
            print(
                "\nStopping here -- no LLM calls made. To generate the article (this makes real, "
                "paid calls to your configured LLM provider), review the command below and then run "
                f"it:\n\n  python -m scripts.run_real_article_pipeline {args.youtube_url!r} "
                f"--episode-id {episode_id} --generate-article\n"
            )
            return

        print("\nEnqueuing article generation (this makes real LLM calls)...")
        await trigger_article_generation(client, args.api_base_url, episode_id)

        print("Waiting for topic analysis -> planning -> section generation -> validation...")
        episode = await wait_for_status(
            client,
            args.api_base_url,
            episode_id,
            ready_statuses=_ARTICLE_READY_STATUSES,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            label="article-generation",
        )
        if episode["status"] == _TERMINAL_FAILURE_STATUS:
            print(f"\nArticle generation FAILED: {episode.get('last_error')}", file=sys.stderr)
            raise SystemExit(1)

        article = await fetch_article(client, args.api_base_url, episode_id)
        print("\n" + format_article_report(article, full_text=args.full_text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("youtube_url", help="A real, publicly accessible YouTube URL")
    parser.add_argument(
        "--api-base-url", default="http://localhost:8000", help="Base URL of the running backend API"
    )
    parser.add_argument(
        "--episode-id", default=None, help="Reuse an already-ingested episode instead of creating one"
    )
    parser.add_argument(
        "--generate-article",
        action="store_true",
        help="Actually enqueue article generation (real, paid LLM calls). Without this flag, the "
        "script stops right after chunking and prints the follow-up command instead.",
    )
    parser.add_argument("--poll-interval", type=float, default=_DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--full-text", action="store_true", help="Print each section's full text instead of a preview"
    )
    args = parser.parse_args()

    try:
        asyncio.run(run(args))
    except TimeoutError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except httpx.HTTPStatusError as exc:
        print(f"\nAPI request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
