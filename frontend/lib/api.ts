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

// Shared by the home page and the blog index: the newest published
// article ("featured") plus the full list, fetched once so both pages
// don't duplicate the same two-call sequence.
export async function getFeaturedAndArticles(): Promise<{
  featured: PublicArticle | null;
  articles: PublicArticleSummary[];
}> {
  const articles = await getPublishedArticles();
  const [latest] = articles;
  const featured = latest ? await getPublishedArticle(latest.episode_id) : null;
  return { featured, articles };
}
