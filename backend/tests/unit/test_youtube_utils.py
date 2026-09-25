import pytest

from app.utils.youtube import InvalidYouTubeURLError, extract_video_id


@pytest.mark.parametrize(
    "url,expected_id",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("http://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=30", "dQw4w9WgXcQ"),
        (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLxyz&index=3&t=42s",
            "dQw4w9WgXcQ",
        ),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("  https://youtu.be/dQw4w9WgXcQ  ", "dQw4w9WgXcQ"),
    ],
)
def test_extracts_video_id_from_valid_urls(url: str, expected_id: str) -> None:
    assert extract_video_id(url) == expected_id


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://vimeo.com/12345678",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch?v=",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/",
        "https://youtu.be/",
        "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
        "javascript:alert(1)",
    ],
)
def test_rejects_invalid_urls(url: str) -> None:
    with pytest.raises(InvalidYouTubeURLError):
        extract_video_id(url)


def test_missing_video_id_is_rejected() -> None:
    with pytest.raises(InvalidYouTubeURLError):
        extract_video_id("https://www.youtube.com/watch?list=PLxyz")
