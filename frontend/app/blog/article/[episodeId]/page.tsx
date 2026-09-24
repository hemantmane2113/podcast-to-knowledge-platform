import Link from "next/link";
import { notFound } from "next/navigation";

const API_BASE_URL = process.env.API_BASE_URL ?? "http://backend:8000";

type PublicSource = {
  start_ms: number;
  end_ms: number;
  text_preview: string;
};

type PublicArticleSection = {
  sequence_number: number;
  heading: string;
  content: string;
  supporting_sources: PublicSource[];
};

type PublicArticle = {
  episode_id: string;
  title: string;
  published_at: string;
  episode_title: string | null;
  channel_name: string | null;
  youtube_url: string;
  thumbnail_url: string | null;
  sections: PublicArticleSection[];
};

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

async function getPublishedArticle(episodeId: string): Promise<PublicArticle | null> {
  const response = await fetch(`${API_BASE_URL}/api/v1/articles/${episodeId}`, { cache: "no-store" });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error("Failed to load article");
  return response.json();
}

export default async function BlogArticlePage({
  params,
}: {
  params: Promise<{ episodeId: string }>;
}) {
  const { episodeId } = await params;
  const article = await getPublishedArticle(episodeId);

  if (!article) {
    notFound();
  }

  return (
    <main className="mx-auto max-w-2xl px-6 py-16">
      <Link href="/blog" className="text-sm text-neutral-500 hover:underline">
        ← All articles
      </Link>

      <h1 className="mt-4 text-3xl font-semibold tracking-tight">{article.title}</h1>
      <p className="mt-2 text-sm text-neutral-500">{formatDate(article.published_at)}</p>

      {(article.episode_title || article.channel_name || article.youtube_url) && (
        <div className="mt-6 rounded-lg border border-neutral-200 bg-neutral-50 px-4 py-3 text-sm text-neutral-600">
          {article.episode_title && (
            <p>
              Based on: <span className="font-medium">{article.episode_title}</span>
            </p>
          )}
          {article.channel_name && <p className="mt-1">Channel: {article.channel_name}</p>}
          {article.youtube_url && (
            <p className="mt-1">
              <a
                href={article.youtube_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-neutral-700 underline hover:text-neutral-900"
              >
                Watch on YouTube →
              </a>
            </p>
          )}
        </div>
      )}

      <article className="mt-10 space-y-10">
        {article.sections.map((section) => (
          <section key={section.sequence_number}>
            <h2 className="text-xl font-semibold">{section.heading}</h2>
            <div className="mt-3 space-y-4 text-base leading-relaxed text-neutral-800">
              {section.content.split("\n\n").map((paragraph, i) => (
                <p key={i}>{paragraph}</p>
              ))}
            </div>
          </section>
        ))}
      </article>
    </main>
  );
}
