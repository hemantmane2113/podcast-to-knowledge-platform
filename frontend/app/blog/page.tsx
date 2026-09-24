import Link from "next/link";

const API_BASE_URL = process.env.API_BASE_URL ?? "http://backend:8000";

type ArticleSummary = {
  episode_id: string;
  title: string;
  published_at: string;
};

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

async function getPublishedArticles(): Promise<ArticleSummary[]> {
  // no-store: this is a live listing of PUBLISHED articles, and it also
  // keeps this route out of static build-time prerendering (there's no
  // backend running during `next build`).
  const response = await fetch(`${API_BASE_URL}/api/v1/articles`, { cache: "no-store" });
  if (!response.ok) return [];
  const body = await response.json();
  return body.articles;
}

export default async function BlogPage() {
  const articles = await getPublishedArticles();

  return (
    <main className="mx-auto max-w-2xl px-6 py-16">
      <h1 className="text-3xl font-semibold tracking-tight">Articles</h1>
      <p className="mt-2 text-neutral-600">Ideas from podcasts, written up for reading.</p>

      {articles.length === 0 ? (
        <p className="mt-10 text-sm text-neutral-500">No articles published yet.</p>
      ) : (
        <ul className="mt-10 space-y-8">
          {articles.map((article) => (
            <li key={article.episode_id} className="border-b border-neutral-200 pb-8">
              <Link
                href={`/blog/article/${article.episode_id}`}
                className="text-xl font-semibold hover:underline"
              >
                {article.title}
              </Link>
              <p className="mt-1 text-sm text-neutral-500">{formatDate(article.published_at)}</p>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
