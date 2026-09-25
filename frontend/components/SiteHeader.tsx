import Link from "next/link";

// Reused, unchanged, across the home page, the article index, and every
// article page -- the editorial masthead + tagline the design brief asks
// for on both "/" and "/blog", kept as one component rather than
// duplicated per page.
//
// "Topics" and "About" have no real route/content behind them yet (only
// "/blog" exists among the requested nav destinations) -- rendered as
// inert labels (no href, aria-disabled) rather than a link to a page
// that doesn't exist, consistent with never inventing content the API
// or the app doesn't actually have.
export function SiteHeader() {
  return (
    <header className="border-b border-rule bg-paper">
      <div className="mx-auto max-w-5xl px-6 py-10 text-center sm:py-14">
        <Link href="/" className="inline-block font-serif text-3xl tracking-tight text-ink sm:text-4xl">
          Conversely
        </Link>
        <p className="mt-2 text-xs font-medium uppercase tracking-[0.2em] text-muted sm:text-sm">
          Long Conversations. Powerful Ideas.
        </p>
        <nav aria-label="Primary" className="mt-6 flex justify-center gap-x-6 gap-y-2 text-sm font-medium">
          <Link href="/blog" className="text-ink-soft transition-colors hover:text-accent">
            Articles
          </Link>
          <span aria-disabled="true" className="cursor-default text-muted/70">
            Topics
          </span>
          <span aria-disabled="true" className="cursor-default text-muted/70">
            About
          </span>
        </nav>
      </div>
    </header>
  );
}
