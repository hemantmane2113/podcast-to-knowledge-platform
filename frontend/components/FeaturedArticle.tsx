import Link from "next/link";
import type { PublicArticle } from "@/lib/api";
import { articleWordCount, firstParagraph, formatDate, readingTimeMinutes } from "@/lib/format";

// The current/latest published article, shown prominently -- every value
// here (title, date, channel, excerpt, reading time) is derived from the
// real article the API returned, never invented.
export function FeaturedArticle({ article }: { article: PublicArticle }) {
  const minutes = readingTimeMinutes(articleWordCount(article));
  const excerpt = article.sections[0] ? firstParagraph(article.sections[0].content) : null;

  return (
    <article className="pb-4">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent">Latest</p>
      <Link href={`/blog/article/${article.episode_id}`} className="group mt-3 block">
        <h2 className="font-serif text-3xl leading-tight text-ink transition-colors group-hover:text-accent sm:text-4xl">
          {article.title}
        </h2>
      </Link>
      {excerpt && <p className="mt-4 max-w-2xl font-serif text-lg leading-relaxed text-ink-soft">{excerpt}</p>}
      <div className="mt-5 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted">
        <span>{formatDate(article.published_at)}</span>
        <span aria-hidden="true">&middot;</span>
        <span>{minutes} min read</span>
        {article.channel_name && (
          <>
            <span aria-hidden="true">&middot;</span>
            <span>{article.channel_name}</span>
          </>
        )}
      </div>
      <Link
        href={`/blog/article/${article.episode_id}`}
        className="mt-6 inline-flex items-center gap-2 text-sm font-semibold text-ink transition-colors hover:text-accent"
      >
        Read the article <span aria-hidden="true">&rarr;</span>
      </Link>
    </article>
  );
}
