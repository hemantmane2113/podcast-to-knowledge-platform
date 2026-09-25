import Link from "next/link";
import { getCoverImage } from "@/lib/cover-images";
import { formatDate } from "@/lib/format";

export type ArticleCardProps = {
  episodeId: string;
  title: string;
  publishedAt: string;
  // Both optional: the list endpoint alone can't supply either (see
  // lib/api.ts::getPublishedArticlesWithDetails), and a "Read Next" card
  // built from summary-only data has neither. Simply omitted when absent
  // -- never a fabricated placeholder value.
  minutes?: number | null;
  category?: string | null;
  className?: string;
};

// The one reusable editorial card: a large image (the article's own
// mapped cover -- see lib/cover-images.ts -- or a neutral placeholder
// when none is mapped yet, never a broken image and never another
// article's or the homepage hero's image), an optional category, the
// title, and reading time/date as subtle metadata. The whole card is a
// single <Link>, not just the title, so clicking anywhere on it
// navigates to the article. Sized to read as a large editorial feature
// (used at ~380-420px wide in ArticleCarousel), not a small thumbnail.
export function ArticleCard({ episodeId, title, publishedAt, minutes, category, className }: ArticleCardProps) {
  const cover = getCoverImage(episodeId);

  return (
    <li className={className}>
      <Link href={`/blog/article/${episodeId}`} className="group block h-full">
        <div className="relative aspect-[4/3] overflow-hidden bg-paper-alt">
          {cover ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={cover.src}
              alt={cover.alt}
              loading="lazy"
              decoding="async"
              draggable={false}
              className="h-full w-full object-cover transition-transform duration-500 ease-out group-hover:scale-[1.03]"
            />
          ) : (
            <div
              aria-hidden="true"
              className="h-full w-full bg-gradient-to-br from-paper-alt via-rule to-paper-alt"
            />
          )}
        </div>

        {category && (
          <p className="mt-5 text-xs font-semibold uppercase tracking-[0.2em] text-accent">{category}</p>
        )}
        <h3
          className={`font-serif text-2xl leading-snug text-ink transition-colors group-hover:text-accent ${
            category ? "mt-2" : "mt-5"
          }`}
        >
          {title}
        </h3>
        <p className="mt-3 text-sm text-muted">
          {formatDate(publishedAt)}
          {minutes ? <> &middot; {minutes} min read</> : null}
        </p>
      </Link>
    </li>
  );
}
