type SourceAttributionProps = {
  episodeTitle: string | null;
  channelName: string | null;
  youtubeUrl: string;
};

// The "SOURCE CONVERSATION" block -- built entirely from Episode's own
// fields as returned by GET /api/v1/articles/{episodeId} (episode_title,
// channel_name, youtube_url); episode_title/channel_name are rendered
// only when the API actually returned them, exactly as the backend's own
// _podcast_attribution_sentence never fabricates a name when absent.
export function SourceAttribution({ episodeTitle, channelName, youtubeUrl }: SourceAttributionProps) {
  return (
    <aside aria-label="Source conversation" className="mt-14 border-y border-rule py-8">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent">Source Conversation</p>
      {episodeTitle && (
        <p className="mt-3 text-base text-ink-soft">
          Based on: <span className="font-medium text-ink">{episodeTitle}</span>
        </p>
      )}
      {channelName && (
        <p className="mt-1 text-base text-ink-soft">
          Channel: <span className="font-medium text-ink">{channelName}</span>
        </p>
      )}
      <p className="mt-4 max-w-2xl text-sm leading-relaxed text-muted">
        This article is based on the original conversation. The ideas, opinions, claims, and perspectives
        belong to the original speakers and source&mdash;not to Conversely.
      </p>
      <a
        href={youtubeUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-ink transition-colors hover:text-accent"
      >
        Watch the full conversation <span aria-hidden="true">&rarr;</span>
      </a>
    </aside>
  );
}
