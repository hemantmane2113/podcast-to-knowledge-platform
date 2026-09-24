import Link from "next/link";
import { notFound } from "next/navigation";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

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

      <article className="mt-10 space-y-10">
        {article.sections.map((section) => (
          <section key={section.sequence_number}>
            <h2 className="text-xl font-semibold">{section.heading}</h2>
            <p className="mt-3 whitespace-pre-wrap text-base leading-relaxed text-neutral-800">
              {section.content}
            </p>
          </section>
        ))}
      </article>
    </main>
  );
}
