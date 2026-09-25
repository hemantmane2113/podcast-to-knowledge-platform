"use client";

import { useEffect, useRef, useState } from "react";
import { ArticleCard } from "./ArticleCard";
import type { CardArticle } from "@/lib/api";

const DRAG_CLICK_THRESHOLD_PX = 6;

// Native overflow-x + CSS scroll-snap -- no carousel dependency. Desktop
// shows ~3 large cards with the next one peeking at the edge and
// functioning arrow controls; touch devices get native swipe scrolling;
// mouse users additionally get press-and-drag scrolling (pointer events),
// since plain overflow-x only responds to a wheel/trackpad gesture, not a
// held mouse-drag, and the brief explicitly asks for both.
export function ArticleCarousel({ articles }: { articles: CardArticle[] }) {
  const trackRef = useRef<HTMLUListElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  const dragState = useRef<{ active: boolean; startX: number; startScrollLeft: number; moved: number } | null>(
    null,
  );
  const [isDragging, setIsDragging] = useState(false);

  const updateScrollState = () => {
    const el = trackRef.current;
    if (!el) return;
    setCanScrollLeft(el.scrollLeft > 8);
    setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 8);
  };

  useEffect(() => {
    updateScrollState();
    window.addEventListener("resize", updateScrollState);
    return () => window.removeEventListener("resize", updateScrollState);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [articles.length]);

  const scroll = (direction: 1 | -1) => {
    const el = trackRef.current;
    if (!el) return;
    el.scrollBy({ left: direction * el.clientWidth * 0.85, behavior: "smooth" });
  };

  // Mouse-only press-and-drag scrolling. Touch is left entirely to the
  // browser's native scroll/snap handling -- adding pointer-drag logic
  // for touch too would fight the OS's own momentum scrolling.
  const onPointerDown = (e: React.PointerEvent<HTMLUListElement>) => {
    if (e.pointerType !== "mouse") return;
    const el = trackRef.current;
    if (!el) return;
    dragState.current = { active: true, startX: e.clientX, startScrollLeft: el.scrollLeft, moved: 0 };
    el.setPointerCapture(e.pointerId);
    setIsDragging(true);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLUListElement>) => {
    const drag = dragState.current;
    const el = trackRef.current;
    if (!drag?.active || !el) return;
    const delta = e.clientX - drag.startX;
    drag.moved = Math.max(drag.moved, Math.abs(delta));
    el.scrollLeft = drag.startScrollLeft - delta;
  };

  const endDrag = (e: React.PointerEvent<HTMLUListElement>) => {
    const drag = dragState.current;
    if (drag?.active) {
      trackRef.current?.releasePointerCapture(e.pointerId);
    }
    dragState.current = drag ? { ...drag, active: false } : null;
    setIsDragging(false);
    updateScrollState();
  };

  // A drag that moved more than a few pixels shouldn't also fire the
  // <Link> navigation underneath the pointer on release. `moved` is
  // consumed (reset to 0) after each click so a stale value from an
  // earlier drag can never suppress a later, unrelated click.
  const onClickCapture = (e: React.MouseEvent<HTMLUListElement>) => {
    const moved = dragState.current?.moved ?? 0;
    if (dragState.current) dragState.current.moved = 0;
    if (moved > DRAG_CLICK_THRESHOLD_PX) {
      e.preventDefault();
      e.stopPropagation();
    }
  };

  return (
    <div className="relative">
      <ul
        ref={trackRef}
        onScroll={updateScrollState}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onClickCapture={onClickCapture}
        className={`flex snap-x snap-mandatory gap-8 overflow-x-auto scroll-smooth pb-2 select-none [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden ${
          isDragging ? "cursor-grabbing scroll-auto" : "cursor-grab"
        }`}
      >
        {articles.map((article) => (
          <ArticleCard
            key={article.episodeId}
            episodeId={article.episodeId}
            title={article.title}
            publishedAt={article.publishedAt}
            minutes={article.minutes}
            category={article.category}
            className="w-[85vw] shrink-0 snap-start sm:w-[360px] lg:w-[400px]"
          />
        ))}
      </ul>

      {canScrollLeft && (
        <button
          type="button"
          onClick={() => scroll(-1)}
          aria-label="Scroll to previous articles"
          className="absolute left-0 top-[38%] hidden -translate-x-4 -translate-y-1/2 rounded-full border border-rule bg-paper p-2.5 text-ink shadow-sm transition-colors hover:text-accent md:flex"
        >
          <ArrowIcon direction="left" />
        </button>
      )}
      {canScrollRight && (
        <button
          type="button"
          onClick={() => scroll(1)}
          aria-label="Scroll to next articles"
          className="absolute right-0 top-[38%] hidden -translate-y-1/2 translate-x-4 rounded-full border border-rule bg-paper p-2.5 text-ink shadow-sm transition-colors hover:text-accent md:flex"
        >
          <ArrowIcon direction="right" />
        </button>
      )}
    </div>
  );
}

function ArrowIcon({ direction }: { direction: "left" | "right" }) {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className="h-5 w-5"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {direction === "left" ? <path d="M15 5 8 12l7 7" /> : <path d="M9 5l7 7-7 7" />}
    </svg>
  );
}
