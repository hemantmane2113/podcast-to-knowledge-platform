// Rendered at the end of every article display, verbatim, per the
// editorial redesign brief -- the same disclaimer text on every article,
// not derived from API data.
export function ArticleDisclaimer() {
  return (
    <section aria-label="A note on attribution" className="mt-10 border-t border-rule pt-8">
      <h2 className="text-xs font-semibold uppercase tracking-[0.2em] text-muted">A note on attribution</h2>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted">
        Conversely transforms long-form conversations and podcasts into readable articles using AI. The
        ideas, opinions, claims, and perspectives discussed in these articles belong to the original
        speakers and sources, not to Conversely. We do not claim ownership of or endorsement of these
        views, nor do we independently verify every claim made in the original conversation.
      </p>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted">
        These articles are intended for informational and educational purposes only and should not be
        treated as professional, medical, financial, legal, or other expert advice. For complete context,
        please refer to the original conversation.
      </p>
    </section>
  );
}
