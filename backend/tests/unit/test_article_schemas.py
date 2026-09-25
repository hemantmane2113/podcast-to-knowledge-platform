"""Tests for app/schemas/article.py's deterministic podcast-attribution
sentence (Live/Draft Article Workflow follow-up fix, requirement #4):
every generated/finalized article must end with a sentence pointing back
to its source podcast, built purely from Episode's own existing metadata
-- never delegated to the LLM, never hard-coding a specific show/host/
guest name so this works identically for every episode."""

import uuid

from app.models.episode import Episode, ProcessingStatus
from app.schemas.article import _podcast_attribution_sentence


def _episode(
    *, channel_name: str | None = None, title: str | None = None, youtube_url: str = "https://youtube.com/watch?v=abc"
) -> Episode:
    return Episode(
        id=uuid.uuid4(),
        youtube_video_id="abc12345678",
        youtube_url=youtube_url,
        status=ProcessingStatus.PUBLISHED,
        title=title,
        channel_name=channel_name,
    )


def test_attribution_uses_channel_name_when_present() -> None:
    episode = _episode(channel_name="The Real Channel", title="The Real Episode Title")
    sentence = _podcast_attribution_sentence(episode)
    assert "The Real Channel" in sentence
    assert episode.youtube_url in sentence
    assert sentence.startswith("This article is based on")


def test_attribution_falls_back_to_title_when_channel_name_is_absent() -> None:
    episode = _episode(channel_name=None, title="The Real Episode Title")
    sentence = _podcast_attribution_sentence(episode)
    assert "The Real Episode Title" in sentence
    assert episode.youtube_url in sentence


def test_attribution_never_fabricates_metadata_that_is_absent() -> None:
    # Neither channel_name nor title populated (ingestion metadata isn't
    # always complete) -- the sentence must still be well-formed and
    # include the URL, without inventing a name.
    episode = _episode(channel_name=None, title=None)
    sentence = _podcast_attribution_sentence(episode)
    assert episode.youtube_url in sentence
    assert "None" not in sentence


def test_attribution_always_includes_the_exact_stored_youtube_url() -> None:
    episode = _episode(
        channel_name="Some Channel", youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
    sentence = _podcast_attribution_sentence(episode)
    assert "https://www.youtube.com/watch?v=dQw4w9WgXcQ" in sentence


def test_attribution_never_hardcodes_a_specific_show_or_person() -> None:
    # Two different episodes must produce two different sentences, driven
    # entirely by Episode's own metadata -- proves nothing is hard-coded.
    episode_a = _episode(channel_name="Channel A")
    episode_b = _episode(channel_name="Channel B")
    assert _podcast_attribution_sentence(episode_a) != _podcast_attribution_sentence(episode_b)
    assert "Channel A" not in _podcast_attribution_sentence(episode_b)
