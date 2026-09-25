import Link from "next/link";

// The homepage hero: masthead text on the left (the page's own <h1>,
// supporting description, CTA, and the brand attribution line -- all
// existing site copy, nothing invented), main.png as the large visual on
// the right. Two columns on larger screens, stacked (text first) on
// mobile. Text and image never overlap -- no copy is ever placed over
// the image itself.
export function HomeHero() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-16 sm:py-20">
      <div className="grid grid-cols-1 items-center gap-12 lg:grid-cols-2 lg:gap-16">
        <div>
          <h1 className="font-serif text-4xl leading-tight text-ink sm:text-5xl lg:text-6xl">
            Long Conversations.
            <br />
            Powerful Ideas.
          </h1>
          <p className="mt-6 max-w-md text-lg leading-relaxed text-ink-soft">
            The ideas, stories, and insights buried inside long-form conversations&mdash;turned into
            thoughtful, readable articles.
          </p>
          <Link
            href="/blog"
            className="mt-8 inline-flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-ink transition-colors hover:text-accent"
          >
            Explore Articles <span aria-hidden="true">&rarr;</span>
          </Link>
          <p className="mt-10 max-w-md font-serif text-base italic leading-relaxed text-muted">
            &ldquo;Ideas belong to the people who expressed them. We turn the conversation into a readable
            story.&rdquo;
          </p>
        </div>

        <div className="relative aspect-[16/9] overflow-hidden border border-rule lg:aspect-[4/3]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/images/articles/main.png" alt="" className="h-full w-full object-cover" />
        </div>
      </div>
    </section>
  );
}
