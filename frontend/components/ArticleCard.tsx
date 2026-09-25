import Link from "next/link";
import { formatDate } from "@/lib/format";

type ArticleCardProps = {
  episodeId: string;
  title: string;
  publishedAt: string;
};

// A plain list entry, not a card -- a title, a date, a thin rule below.
// Matches the "no excessive cards" design direction; renders as an <li>
// so callers place it inside a <ul>.
export function ArticleCard({ episodeId, title, publishedAt }: ArticleCardProps) {
  return (
    <li className="border-b border-rule py-8 first:pt-0 last:border-b-0">
      <Link href={`/blog/article/${episodeId}`} className="group block">
        <h3 className="font-serif text-2xl leading-snug text-ink transition-colors group-hover:text-accent">
          {title}
        </h3>
      </Link>
      <p className="mt-2 text-sm text-muted">{formatDate(publishedAt)}</p>
    </li>
  );
}
