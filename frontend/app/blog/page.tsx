import { ArticleCard } from "@/components/ArticleCard";
import { EditorialSection } from "@/components/EditorialSection";
import { FeaturedArticle } from "@/components/FeaturedArticle";
import { SiteFooter } from "@/components/SiteFooter";
import { SiteHeader } from "@/components/SiteHeader";
import { getFeaturedAndArticles } from "@/lib/api";

export default async function BlogPage() {
  const { featured, articles } = await getFeaturedAndArticles();
  const rest = articles.slice(1);

  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <main className="flex-1">
        <div className="mx-auto max-w-3xl px-6 py-16">
          <h1 className="font-serif text-3xl text-ink">Articles</h1>
          <p className="mt-2 text-ink-soft">Ideas from long conversations, written up for reading.</p>

          {articles.length === 0 ? (
            <p className="mt-16 border-t border-rule pt-16 text-center text-sm text-muted">
              No articles published yet.
            </p>
          ) : (
            <>
              <EditorialSection eyebrow="Featured" className="mt-12 border-t border-rule pt-12">
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
                <EditorialSection eyebrow="Latest Articles" className="mt-16">
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
