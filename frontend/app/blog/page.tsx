import { ArticleCarousel } from "@/components/ArticleCarousel";
import { EditorialSection } from "@/components/EditorialSection";
import { SiteFooter } from "@/components/SiteFooter";
import { SiteHeader } from "@/components/SiteHeader";
import { getPublishedArticlesWithDetails } from "@/lib/api";
import { toCardArticle } from "@/lib/format";

export default async function BlogPage() {
  const articles = await getPublishedArticlesWithDetails();
  const cards = articles.map(toCardArticle);

  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <main className="flex-1">
        <div className="mx-auto max-w-3xl px-6 py-16">
          <h1 className="font-serif text-3xl text-ink">Articles</h1>
          <p className="mt-2 text-ink-soft">Ideas from long conversations, written up for reading.</p>
        </div>

        {cards.length === 0 ? (
          <p className="mx-auto max-w-3xl border-t border-rule px-6 py-16 text-center text-sm text-muted">
            No articles published yet.
          </p>
        ) : (
          <div className="border-t border-rule py-16">
            <div className="mx-auto max-w-6xl px-6">
              <EditorialSection eyebrow="Every Conversation">
                <ArticleCarousel articles={cards} />
              </EditorialSection>
            </div>
          </div>
        )}
      </main>
      <SiteFooter />
    </div>
  );
}
