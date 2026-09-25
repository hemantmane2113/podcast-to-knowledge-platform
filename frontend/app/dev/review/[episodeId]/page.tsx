"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Episode = {
  id: string;
  status: string;
  title: string | null;
  last_error: string | null;
};

// Episode.status Decoupling: article-generation progress lives on
// ProcessingJob.status now, never on Episode.status -- this mirrors
// app/schemas/article.py::ArticleGenerationStatusResponse.
type GenerationStatus = {
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | null;
  error_message: string | null;
};

type SourceChunk = {
  id: string;
  start_ms: number;
  end_ms: number;
  text_preview: string;
};

type ArticleSection = {
  sequence_number: number;
  heading: string;
  content: string;
  supporting_chunks: SourceChunk[];
};

type ValidationCheck = {
  name: string;
  passed: boolean;
  details: string;
};

type ValidationResult = {
  passed: boolean;
  checks: ValidationCheck[];
  created_at: string;
};

type Article = {
  id: string;
  title: string;
  revision_count: number;
  episode_status: string;
  sections: ArticleSection[];
  latest_validation: ValidationResult | null;
};

function formatTimestamp(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

// This is a minimal human-review surface (Phase I) -- not the real
// editorial CMS (that's a later phase). Live/Draft Article Workflow: this
// page always shows the episode's DRAFT article (GET .../article?draft=true)
// -- the in-progress/awaiting-review generation -- never the live one, so a
// reviewer can judge a freshly generated draft (including a regeneration of
// an already-published episode) without it ever being visible publicly.
// "Promote to Live" is the explicit, deliberate step that makes it public.
export default function ReviewPage() {
  const params = useParams<{ episodeId: string }>();
  const episodeId = params.episodeId;

  const [episode, setEpisode] = useState<Episode | null>(null);
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus | null>(null);
  const [article, setArticle] = useState<Article | null>(null);
  const [articleError, setArticleError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [promoting, setPromoting] = useState(false);
  const [promoteError, setPromoteError] = useState<string | null>(null);
  const [regenerating, setRegenerating] = useState(false);
  const [regenerateError, setRegenerateError] = useState<string | null>(null);

  // Episode.status Decoupling: generation progress lives on
  // ProcessingJob.status now (via generation-status), never on
  // Episode.status. Computed once, before any early return, so both the
  // poll effect and the Regenerate button's disabled state agree.
  const inFlight = generationStatus?.status === "PENDING" || generationStatus?.status === "RUNNING";

  async function refresh() {
    const episodeResponse = await fetch(`${API_BASE_URL}/api/v1/episodes/${episodeId}`);
    if (episodeResponse.ok) setEpisode(await episodeResponse.json());

    const statusResponse = await fetch(`${API_BASE_URL}/api/v1/episodes/${episodeId}/generation-status`);
    if (statusResponse.ok) setGenerationStatus(await statusResponse.json());

    const articleResponse = await fetch(
      `${API_BASE_URL}/api/v1/episodes/${episodeId}/article?draft=true`
    );
    if (articleResponse.ok) {
      setArticle(await articleResponse.json());
      setArticleError(null);
    } else if (articleResponse.status === 404) {
      setArticle(null);
      setArticleError(null); // no draft yet -- not an error state
    } else {
      const body = await articleResponse.json().catch(() => null);
      setArticleError(body?.message ?? "Failed to load draft article");
    }
  }

  useEffect(() => {
    if (!episodeId) return;
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [episodeId]);

  useEffect(() => {
    // Poll while the latest ARTICLE_GENERATION job is still
    // PENDING/RUNNING -- this also covers the job a Regenerate click just
    // enqueued, since that flips generationStatus back to PENDING on the
    // next refresh() below and re-triggers this effect.
    if (!inFlight) return;
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inFlight]);

  async function handleGenerate() {
    setGenerating(true);
    try {
      await fetch(`${API_BASE_URL}/api/v1/episodes/${episodeId}/generate-article`, {
        method: "POST",
      });
      await refresh();
    } finally {
      setGenerating(false);
    }
  }

  async function handleRegenerate() {
    // Uses the existing explicit regeneration endpoint
    // (POST /episodes/{id}/regenerate-article -- ArticleService
    // .request_regeneration) exactly as it already works: it only ever
    // touches the episode's DRAFT plan/article (replacing a stale/failed/
    // partial one, if any) and enqueues a fresh ARTICLE_GENERATION job.
    // The live article is never read, deleted, or otherwise touched here
    // or by that endpoint.
    setRegenerating(true);
    setRegenerateError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/episodes/${episodeId}/regenerate-article`, {
        method: "POST",
      });
      if (response.ok) {
        await refresh(); // picks up the new draft state and the freshly PENDING job, resuming polling
      } else {
        const body = await response.json().catch(() => null);
        setRegenerateError(body?.message ?? "Failed to start regeneration");
      }
    } finally {
      setRegenerating(false);
    }
  }

  async function handlePromote() {
    setPromoting(true);
    setPromoteError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/episodes/${episodeId}/promote-draft`, {
        method: "POST",
      });
      if (response.ok) {
        await refresh();
      } else {
        const body = await response.json().catch(() => null);
        setPromoteError(body?.message ?? "Failed to promote draft");
      }
    } finally {
      setPromoting(false);
    }
  }

  if (!episode) {
    return (
      <main className="mx-auto max-w-3xl px-6 py-16">
        <p className="text-sm text-neutral-500">Loading episode…</p>
      </main>
    );
  }

  const canPromote = generationStatus?.status === "COMPLETED" && article?.latest_validation?.passed;

  return (
    <main className="mx-auto max-w-3xl px-6 py-16">
      <p className="text-sm text-neutral-500">Dev: Article Review — not the real admin UI</p>
      <h1 className="mt-1 text-2xl font-semibold">{episode.title ?? episode.id}</h1>

      <dl className="mt-4 space-y-1 text-sm">
        <div className="flex gap-2">
          <dt className="text-neutral-500">Episode status</dt>
          <dd className="font-mono">{episode.status}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="text-neutral-500">Draft generation</dt>
          <dd className="font-mono">{generationStatus?.status ?? "—"}</dd>
        </div>
        {episode.last_error && (
          <div className="flex gap-2">
            <dt className="text-neutral-500">Error</dt>
            <dd className="text-red-600">{episode.last_error}</dd>
          </div>
        )}
        {generationStatus?.error_message && (
          <div className="flex gap-2">
            <dt className="text-neutral-500">Generation error</dt>
            <dd className="text-red-600">{generationStatus.error_message}</dd>
          </div>
        )}
      </dl>

      {!article && (
        <button
          onClick={handleGenerate}
          disabled={generating}
          className="mt-6 rounded bg-neutral-900 px-4 py-2 text-sm text-white disabled:opacity-50"
        >
          {generating ? "Starting…" : "Generate Article"}
        </button>
      )}

      {article && (
        <div className="mt-6 flex gap-2">
          <button
            onClick={handlePromote}
            disabled={!canPromote || promoting}
            className="rounded bg-neutral-900 px-4 py-2 text-sm text-white disabled:opacity-50"
          >
            {promoting ? "Promoting…" : "Promote to Live"}
          </button>
          <button
            onClick={handleRegenerate}
            disabled={inFlight || regenerating}
            className="rounded border border-neutral-900 px-4 py-2 text-sm text-neutral-900 disabled:opacity-50"
          >
            {regenerating ? "Starting…" : "Regenerate"}
          </button>
        </div>
      )}
      {promoteError && <p className="mt-2 text-sm text-red-600">{promoteError}</p>}
      {regenerateError && <p className="mt-2 text-sm text-red-600">{regenerateError}</p>}

      {articleError && <p className="mt-4 text-sm text-red-600">{articleError}</p>}

      {article && (
        <section className="mt-10">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Validation</h2>
            <span
              className={`rounded px-2 py-1 text-xs font-medium ${
                article.latest_validation?.passed
                  ? "bg-green-100 text-green-800"
                  : "bg-amber-100 text-amber-800"
              }`}
            >
              {article.latest_validation?.passed ? "Passed" : "Needs review"}
            </span>
          </div>
          <p className="mt-1 text-xs text-neutral-500">Revision attempts: {article.revision_count}</p>
          {article.latest_validation && (
            <ul className="mt-3 space-y-1 text-sm">
              {article.latest_validation.checks.map((check) => (
                <li key={check.name} className="flex gap-2">
                  <span className={check.passed ? "text-green-700" : "text-red-600"}>
                    {check.passed ? "✓" : "✗"}
                  </span>
                  <span>
                    <span className="font-medium">{check.name}</span>
                    {": "}
                    <span className="text-neutral-600">{check.details}</span>
                  </span>
                </li>
              ))}
            </ul>
          )}

          <h2 className="mt-10 text-lg font-semibold">{article.title}</h2>
          <div className="mt-4 space-y-8">
            {article.sections.map((section) => (
              <article key={section.sequence_number}>
                <h3 className="text-base font-semibold">{section.heading}</h3>
                <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-neutral-800">
                  {section.content}
                </p>
                {section.supporting_chunks.length > 0 && (
                  <div className="mt-3 rounded border border-neutral-200 p-3">
                    <p className="text-xs font-medium text-neutral-500">Sources</p>
                    <ul className="mt-1 space-y-1">
                      {section.supporting_chunks.map((chunk) => (
                        <li key={chunk.id} className="text-xs text-neutral-600">
                          <span className="font-mono">
                            {formatTimestamp(chunk.start_ms)}–{formatTimestamp(chunk.end_ms)}
                          </span>{" "}
                          {chunk.text_preview}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </article>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
