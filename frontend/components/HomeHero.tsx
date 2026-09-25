import Link from "next/link";

// The homepage hero: masthead text on the left (the page's own <h1>,
// supporting description, CTA, and the brand attribution line -- all
// existing site copy, nothing invented), main.png as the large visual on
// the right. Two columns on larger screens, stacked (text first) on
// mobile. Text and image never overlap -- no copy is ever placed over
// the image itself.
//
// Image container: a single wide aspect-[16/9] at every breakpoint (no
// narrower override on desktop) so main.png reads as a clearly wide,
// panoramic editorial image rather than a near-square card -- 16/9 is
// also closer to main.png's own very wide native aspect than a narrower
// ratio would be, so less of the image needs to be cropped away by
// object-cover. The desktop grid is asymmetric (5fr text / 7fr image,
// not an even 50/50 split) specifically so the wider image gets more
// horizontal space to actually show that width, not just a wide aspect
// ratio squeezed into a half-width column.
export function HomeHero() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-16 sm:py-20">
      <div className="grid grid-cols-1 items-center gap-12 lg:grid-cols-[5fr_7fr] lg:gap-16">
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

        <div className="relative aspect-[16/9] overflow-hidden border border-rule">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/images/articles/main.png" alt="" className="h-full w-full object-cover" />
        </div>
      </div>
    </section>
  );
}
