import Link from "next/link";
import { notFound } from "next/navigation";
import { ArticleCard } from "@/components/ArticleCard";
import { ArticleDisclaimer } from "@/components/ArticleDisclaimer";
import { SiteFooter } from "@/components/SiteFooter";
import { SiteHeader } from "@/components/SiteHeader";
import { SourceAttribution } from "@/components/SourceAttribution";
import { getPublishedArticle, getPublishedArticles } from "@/lib/api";
import { articleWordCount, formatDate, readingTimeMinutes } from "@/lib/format";

export default async function BlogArticlePage({
  params,
}: {
  params: Promise<{ episodeId: string }>;
}) {
  const { episodeId } = await params;
  const [article, allArticles] = await Promise.all([getPublishedArticle(episodeId), getPublishedArticles()]);

  if (!article) {
    notFound();
  }

  const minutes = readingTimeMinutes(articleWordCount(article));
  const nextArticle = allArticles.find((a) => a.episode_id !== article.episode_id) ?? null;

  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <main className="flex-1">
        <article className="mx-auto max-w-2xl px-6 py-16">
          <nav aria-label="Breadcrumb">
            <Link href="/blog" className="text-sm text-muted transition-colors hover:text-accent">
              &larr; All articles
            </Link>
          </nav>

          <p className="mt-6 text-sm text-muted">
            {formatDate(article.published_at)} &middot; {minutes} min read
          </p>
          <h1 className="mt-3 font-serif text-4xl leading-tight text-ink sm:text-5xl">{article.title}</h1>

          {(article.channel_name || article.episode_title) && (
            <p className="mt-4 text-sm text-ink-soft">
              Based on a conversation with{" "}
              <span className="font-medium text-ink">{article.channel_name ?? article.episode_title}</span>
            </p>
          )}

          <div className="mt-10 space-y-10">
            {article.sections.map((section, sectionIndex) => (
              <section key={section.sequence_number}>
                <h2 className="font-serif text-xl text-ink">{section.heading}</h2>
                <div className="mt-3 space-y-5">
                  {section.content.split("\n\n").map((paragraph, paragraphIndex) => (
                    <p
                      key={paragraphIndex}
                      className={
                        sectionIndex === 0 && paragraphIndex === 0
                          ? "font-serif text-xl leading-relaxed text-ink-soft"
                          : "font-serif text-lg leading-relaxed text-ink-soft"
                      }
                    >
                      {paragraph}
                    </p>
                  ))}
                </div>
              </section>
            ))}
          </div>

          <SourceAttribution
            episodeTitle={article.episode_title}
            channelName={article.channel_name}
            youtubeUrl={article.youtube_url}
          />

          <ArticleDisclaimer />

          {nextArticle && (
            <div className="mt-14 border-t border-rule pt-10">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent">Read Next</p>
              <ul className="mt-4">
                <ArticleCard
                  episodeId={nextArticle.episode_id}
                  title={nextArticle.title}
                  publishedAt={nextArticle.published_at}
                />
              </ul>
            </div>
          )}
        </article>
      </main>
      <SiteFooter />
    </div>
  );
}
