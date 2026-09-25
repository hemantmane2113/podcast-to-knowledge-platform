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

        <div className="mt-8 max-w-2xl">
          <h2 className="text-xs font-semibold uppercase tracking-[0.2em] text-muted">Editorial Note</h2>
          <p className="mt-3 text-xs leading-relaxed text-muted">
            Conversely transforms long-form conversations and podcasts into readable articles using AI.
            The ideas, opinions, claims, and perspectives discussed in these articles belong to the
            original speakers and sources, not to Conversely. We do not claim ownership of or endorsement
            of these views, nor do we independently verify every claim made in the original conversation.
          </p>
          <p className="mt-3 text-xs leading-relaxed text-muted">
            These articles are intended for informational and educational purposes only and should not be
            treated as professional, medical, financial, legal, or other expert advice. For complete
            context, please refer to the original conversation.
          </p>
        </div>
      </div>
    </footer>
  );
}
