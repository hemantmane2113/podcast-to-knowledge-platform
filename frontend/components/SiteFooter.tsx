import Link from "next/link";

const INERT_FOOTER_LINKS = ["Topics", "About", "How It Works", "Disclaimer", "Privacy", "Terms", "Contact"];

// Same "no route yet" treatment as SiteHeader's Topics/About: rendered as
// inert labels, not links to pages that don't exist. Only "Articles"
// points somewhere real.
export function SiteFooter() {
  return (
    <footer className="border-t border-rule bg-paper-alt">
      <div className="mx-auto max-w-5xl px-6 py-16">
        <div className="font-serif text-2xl text-ink">Conversely</div>
        <p className="mt-1 text-xs font-medium uppercase tracking-[0.2em] text-muted">
          Long Conversations. Powerful Ideas.
        </p>
        <p className="mt-5 max-w-xl font-serif text-lg italic leading-relaxed text-ink-soft">
          &ldquo;Ideas belong to the people who expressed them. We turn the conversation into a readable
          story.&rdquo;
        </p>

        <nav aria-label="Footer" className="mt-8 flex flex-wrap gap-x-6 gap-y-2 text-sm">
          <Link href="/blog" className="text-ink-soft transition-colors hover:text-accent">
            Articles
          </Link>
          {INERT_FOOTER_LINKS.map((label) => (
            <span key={label} aria-disabled="true" className="cursor-default text-muted/70">
              {label}
            </span>
          ))}
        </nav>

        <p className="mt-8 max-w-2xl text-xs leading-relaxed text-muted">
          Conversely turns long-form conversations into readable essays. We don&apos;t claim the ideas as
          our own. Articles are AI-generated summaries of source conversations and may not capture every
          nuance or qualification. Please consult the original source for full context.
        </p>
      </div>
    </footer>
  );
}
