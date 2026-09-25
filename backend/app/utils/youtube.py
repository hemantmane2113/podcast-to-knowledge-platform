import re
from urllib.parse import parse_qs, urlparse

from app.core.exceptions import AppError

# A YouTube video ID is 11 characters of A-Z a-z 0-9 _ -. This is documented
# platform behavior, not an assumption we're inventing.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

_ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}


class InvalidYouTubeURLError(AppError, ValueError):
    """Raised when a string isn't a recognizable YouTube video URL."""

    code = "INVALID_YOUTUBE_URL"
    status_code = 400

    def __init__(self, url: str):
        self.url = url
        super().__init__(f"Not a valid YouTube video URL: {url!r}")


def extract_video_id(url: str) -> str:
    """Extract the 11-character YouTube video ID from a URL.

    Supports the common forms:
      https://www.youtube.com/watch?v=VIDEO_ID
      https://youtube.com/watch?v=VIDEO_ID
      https://youtu.be/VIDEO_ID
      https://m.youtube.com/watch?v=VIDEO_ID
      https://www.youtube.com/shorts/VIDEO_ID
      https://www.youtube.com/embed/VIDEO_ID

    Extra query parameters (timestamps, playlist info, etc.) are ignored.
    Raises InvalidYouTubeURLError for anything else — arbitrary URLs are
    never assumed valid.
    """
    if not url or not isinstance(url, str):
        raise InvalidYouTubeURLError(url)

    try:
        parsed = urlparse(url.strip())
    except ValueError as exc:
        raise InvalidYouTubeURLError(url) from exc

    if parsed.scheme not in ("http", "https"):
        raise InvalidYouTubeURLError(url)

    host = parsed.netloc.lower()
    if host not in _ALLOWED_HOSTS:
        raise InvalidYouTubeURLError(url)

    video_id: str | None = None

    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0] or None
    else:
        path_segments = [segment for segment in parsed.path.split("/") if segment]
        if parsed.path == "/watch" or (path_segments and path_segments[0] == "watch"):
            query = parse_qs(parsed.query)
            values = query.get("v")
            video_id = values[0] if values else None
        elif path_segments and path_segments[0] in ("shorts", "embed", "live"):
            video_id = path_segments[1] if len(path_segments) > 1 else None

    if not video_id or not _VIDEO_ID_RE.match(video_id):
        raise InvalidYouTubeURLError(url)

    return video_id
