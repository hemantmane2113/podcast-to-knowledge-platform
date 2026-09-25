import Link from "next/link";
import { ArticleCard } from "@/components/ArticleCard";
import { EditorialSection } from "@/components/EditorialSection";
import { FeaturedArticle } from "@/components/FeaturedArticle";
import { SiteFooter } from "@/components/SiteFooter";
import { SiteHeader } from "@/components/SiteHeader";
import { getFeaturedAndArticles } from "@/lib/api";

export default async function HomePage() {
  const { featured, articles } = await getFeaturedAndArticles();
  const rest = articles.slice(1);

  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <main className="flex-1">
        <section className="mx-auto max-w-3xl px-6 py-16 text-center sm:py-20">
          <p className="text-lg leading-relaxed text-ink-soft sm:text-xl">
            The ideas, stories, and insights buried inside long-form conversations&mdash;turned into
            thoughtful, readable articles.
          </p>
          <p className="mt-6 font-serif text-xl italic leading-relaxed text-ink sm:text-2xl">
            &ldquo;Ideas belong to the people who expressed them. We turn the conversation into a readable
            story.&rdquo;
          </p>
          <Link
            href="/blog"
            className="mt-8 inline-flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-ink transition-colors hover:text-accent"
          >
            Explore Articles <span aria-hidden="true">&rarr;</span>
          </Link>
        </section>

        <div className="mx-auto max-w-3xl px-6 pb-24">
          {articles.length === 0 ? (
            <p className="border-t border-rule py-16 text-center text-sm text-muted">
              No articles published yet. Check back soon.
            </p>
          ) : (
            <>
              <EditorialSection eyebrow="Featured" className="border-t border-rule pt-12">
                {featured ? (
                  <FeaturedArticle article={featured} />
                ) : (
                  <ul>
                    <ArticleCard
                      episodeId={articles[0].episode_id}
                      title={articles[0].title}
                      publishedAt={articles[0].published_at}
                    />
                  </ul>
                )}
              </EditorialSection>

              {rest.length > 0 && (
                <EditorialSection eyebrow="Latest Ideas" className="mt-16">
                  <ul>
                    {rest.map((article) => (
                      <ArticleCard
                        key={article.episode_id}
                        episodeId={article.episode_id}
                        title={article.title}
                        publishedAt={article.published_at}
                      />
                    ))}
                  </ul>
                </EditorialSection>
              )}
            </>
          )}
        </div>
      </main>
      <SiteFooter />
    </div>
  );
}
