// Client for the real, existing public article API
// (GET /api/v1/articles, GET /api/v1/articles/{episodeId}) --
// see backend/app/schemas/article.py::PublicArticleListResponse /
// PublicArticleResponse for the exact response shapes these types mirror.
// No field here is invented: every property matches what the backend
// actually returns.

const API_BASE_URL = process.env.API_BASE_URL ?? "http://backend:8000";

export type PublicArticleSummary = {
  episode_id: string;
  title: string;
  published_at: string;
};

export type PublicSource = {
  start_ms: number;
  end_ms: number;
  text_preview: string;
};

export type PublicArticleSection = {
  sequence_number: number;
  heading: string;
  content: string;
  supporting_sources: PublicSource[];
};

export type PublicArticle = {
  episode_id: string;
  title: string;
  published_at: string;
  episode_title: string | null;
  channel_name: string | null;
  youtube_url: string;
  thumbnail_url: string | null;
  sections: PublicArticleSection[];
};

export async function getPublishedArticles(): Promise<PublicArticleSummary[]> {
  // no-store: a live listing of PUBLISHED articles, and keeps this route
  // out of static build-time prerendering (there's no backend running
  // during `next build`).
  const response = await fetch(`${API_BASE_URL}/api/v1/articles`, { cache: "no-store" });
  if (!response.ok) return [];
  const body = await response.json();
  return body.articles;
}

export async function getPublishedArticle(episodeId: string): Promise<PublicArticle | null> {
  const response = await fetch(`${API_BASE_URL}/api/v1/articles/${episodeId}`, { cache: "no-store" });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error("Failed to load article");
  return response.json();
}

// The list endpoint only returns id/title/published_at (see
// PublicArticleSummaryResponse) -- no reading time, no channel/category.
// The editorial card system (ArticleCard) needs both, so the discovery
// pages fetch each article's own real detail response too, rather than
// ever inventing a reading time or category. Fine at the catalog size
// this product has today; revisit (e.g. a dedicated summary field on the
// backend) if the published catalog grows large enough for N+1 detail
// fetches to matter.
export async function getPublishedArticlesWithDetails(): Promise<PublicArticle[]> {
  const summaries = await getPublishedArticles();
  const details = await Promise.all(summaries.map((s) => getPublishedArticle(s.episode_id)));
  return details.filter((article): article is PublicArticle => article !== null);
}

// The exact, flat shape ArticleCard renders -- derived entirely from a
// real PublicArticle, never fabricated.
export type CardArticle = {
  episodeId: string;
  title: string;
  publishedAt: string;
  minutes: number | null;
  category: string | null;
};
