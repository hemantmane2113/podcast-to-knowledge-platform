import Link from "next/link";
import { ArticleCarousel } from "@/components/ArticleCarousel";
import { EditorialSection } from "@/components/EditorialSection";
import { HomeHero } from "@/components/HomeHero";
import { SiteFooter } from "@/components/SiteFooter";
import { SiteHeader } from "@/components/SiteHeader";
import { getPublishedArticlesWithDetails } from "@/lib/api";
import { toCardArticle } from "@/lib/format";

export default async function HomePage() {
  const articles = await getPublishedArticlesWithDetails();
  const cards = articles.map(toCardArticle);

  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <main className="flex-1">
        <HomeHero />

        {cards.length === 0 ? (
          <p className="mx-auto max-w-3xl border-t border-rule px-6 py-16 text-center text-sm text-muted">
            No articles published yet. Check back soon.
          </p>
        ) : (
          <>
            <div className="border-t border-rule py-16">
              <div className="mx-auto max-w-6xl px-6">
                <EditorialSection eyebrow="Latest Ideas">
                  <ArticleCarousel articles={cards} />
                </EditorialSection>
              </div>
            </div>

            <div className="mx-auto max-w-3xl px-6 pb-16 text-center">
              <p className="text-ink-soft">Every published article, all in one place.</p>
              <Link
                href="/blog"
                className="mt-4 inline-flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-ink transition-colors hover:text-accent"
              >
                More conversations <span aria-hidden="true">&rarr;</span>
              </Link>
            </div>
          </>
        )}
      </main>
      <SiteFooter />
    </div>
  );
}
