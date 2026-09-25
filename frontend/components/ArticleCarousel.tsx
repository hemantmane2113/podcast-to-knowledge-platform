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
  // browser's native scroll/snap handling.
  //
  // This block fixes two real, confirmed bugs (found via a real React/
  // Next.js click-through and drag test, not just reasoned about --
  // static-HTML mockups of this logic hid both):
  //
  // 1. Drag stopped moving after ~1 pixel. Chromium was firing a native
  //    `pointercancel` almost immediately after pointerdown on this
  //    `overflow-x: auto` element -- confirmed directly with a bare
  //    document-level pointercancel listener -- handing the gesture off
  //    to its own native scroll/pan handling and permanently ending the
  //    JS pointer event stream (no further pointermove/pointerup ever
  //    arrived). The fix is the standard one: call `e.preventDefault()`
  //    on pointerdown AND on every pointermove of an active drag, which
  //    tells the browser this gesture is being fully handled by JS. Both
  //    calls are needed -- removing either one brought the cancellation
  //    back in testing. Scoped to mouse only (the early return below), so
  //    touch's native panning is never affected by either call.
  //
  // 2. Even once dragging moved the mouse correctly, `el.scrollLeft =
  //    ...` assignments were being instantly reverted back to the
  //    nearest snap point -- confirmed by reading the property back
  //    immediately after setting it. CSS scroll-snap re-snaps a raw
  //    property assignment far more aggressively than an actual
  //    (trackpad/touch) scroll gesture, since the browser doesn't treat
  //    it as a real user scroll. Fixed by dropping the snap classes
  //    entirely while `isDragging` is true, restoring them only once the
  //    drag ends -- which is also what gives the drag a satisfying snap
  //    to the nearest card on release.
  //
  // (A previous attempt also tried calling setPointerCapture, either
  // unconditionally on pointerdown or lazily once real movement was
  // detected -- both made things worse: the former made Chromium
  // retarget the eventual `click` event to this <ul> instead of the
  // actually-clicked card, so navigating to an article never fired at
  // all; the latter still hit the pointercancel issue above. Pointer
  // capture is not used here at all.)
  const onPointerDown = (e: React.PointerEvent<HTMLUListElement>) => {
    if (e.pointerType !== "mouse") return;
    const el = trackRef.current;
    if (!el) return;
    e.preventDefault();
    dragState.current = { active: true, startX: e.clientX, startScrollLeft: el.scrollLeft, moved: 0 };
    setIsDragging(true);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLUListElement>) => {
    const drag = dragState.current;
    const el = trackRef.current;
    if (!drag?.active || !el) return;
    e.preventDefault();
    const delta = e.clientX - drag.startX;
    drag.moved = Math.max(drag.moved, Math.abs(delta));
    el.scrollLeft = drag.startScrollLeft - delta;
  };

  const endDrag = () => {
    const drag = dragState.current;
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
        className={`flex gap-8 overflow-x-auto pb-2 select-none [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden ${
          isDragging ? "cursor-grabbing snap-none scroll-auto" : "cursor-grab snap-x snap-mandatory scroll-smooth"
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
