// Deterministic, frontend-only episode_id -> cover image mapping.
// The backend/database have no image field (Episode.thumbnail_url is a
// separate, existing field -- see PublicArticleResponse -- but it's the
// literal video/podcast thumbnail, which the design brief explicitly
// rules out: "no podcast thumbnail"). Rather than touch the backend or
// database, editorial cover art lives entirely here, keyed by the real
// episode_id the API already returns.
//
// main.png (frontend/public/images/articles/main.png) is reserved for
// the homepage hero only (see components/HomeHero.tsx) -- explicitly
// never a card image, per the brief -- so it is not part of this map and
// is never returned as a fallback here.
export type CoverImage = {
  src: string;
  alt: string;
};

const COVER_IMAGES: Record<string, CoverImage> = {
  "bae390d7-12a7-4524-b829-7596346c9b64": {
    src: "/images/articles/card1.png",
    alt: "",
  },
};

// null for any article without its own entry -- ArticleCard renders a
// neutral placeholder rather than a broken image or a borrowed image
// that belongs to a different article/context.
export function getCoverImage(episodeId: string): CoverImage | null {
  return COVER_IMAGES[episodeId] ?? null;
}
