import type { CardArticle, PublicArticle } from "./api";

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

const WORDS_PER_MINUTE = 200;

// Derived from the article's own real content -- never a fabricated or
// hardcoded figure.
export function articleWordCount(article: Pick<PublicArticle, "sections">): number {
  return article.sections.reduce(
    (total, section) => total + section.content.trim().split(/\s+/).filter(Boolean).length,
    0,
  );
}

export function readingTimeMinutes(wordCount: number): number {
  return Math.max(1, Math.round(wordCount / WORDS_PER_MINUTE));
}

// The first paragraph of a section's real content, lightly truncated for
// a card/featured excerpt -- never invented copy, just a slice of the
// actual generated article text.
export function firstParagraph(content: string, maxChars = 240): string {
  const [first] = content.split("\n\n");
  const clean = (first ?? "").trim();
  if (clean.length <= maxChars) return clean;
  return `${clean.slice(0, maxChars).replace(/\s+\S*$/, "")}…`;
}

// ArticleCard's props, derived from a real PublicArticle -- `category`
// uses channel_name (the closest real categorization signal the API
// provides; there is no dedicated topic/category field), only ever shown
// when the API actually returned one.
export function toCardArticle(article: PublicArticle): CardArticle {
  return {
    episodeId: article.episode_id,
    title: article.title,
    publishedAt: article.published_at,
    minutes: readingTimeMinutes(articleWordCount(article)),
    category: article.channel_name,
  };
}
