import type { ReactNode } from "react";

type EditorialSectionProps = {
  eyebrow?: string;
  title?: string;
  className?: string;
  children: ReactNode;
};

// A small, reusable "labeled section" wrapper -- used for FEATURED /
// LATEST IDEAS on the home page and the equivalent sections on the blog
// index, so that layout isn't duplicated per page.
export function EditorialSection({ eyebrow, title, className, children }: EditorialSectionProps) {
  return (
    <section className={className}>
      {(eyebrow || title) && (
        <div className="mb-8">
          {eyebrow && <p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent">{eyebrow}</p>}
          {title && <h2 className="mt-1 font-serif text-2xl text-ink">{title}</h2>}
        </div>
      )}
      {children}
    </section>
  );
}
