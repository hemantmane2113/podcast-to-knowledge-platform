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
// editorial CMS (that's a later phase). It shows the generated article,
// its sections' supporting source chunks with timestamps, processing
// status, and validation results so a reviewer can judge the output
// before anything gets published.
export default function ReviewPage() {
  const params = useParams<{ episodeId: string }>();
  const episodeId = params.episodeId;

  const [episode, setEpisode] = useState<Episode | null>(null);
  const [article, setArticle] = useState<Article | null>(null);
  const [articleError, setArticleError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  async function refresh() {
    const episodeResponse = await fetch(`${API_BASE_URL}/api/v1/episodes/${episodeId}`);
    if (episodeResponse.ok) setEpisode(await episodeResponse.json());

    const articleResponse = await fetch(`${API_BASE_URL}/api/v1/episodes/${episodeId}/article`);
    if (articleResponse.ok) {
      setArticle(await articleResponse.json());
      setArticleError(null);
    } else if (articleResponse.status === 404) {
      setArticle(null);
      setArticleError(null); // no article yet -- not an error state
    } else {
      const body = await articleResponse.json().catch(() => null);
      setArticleError(body?.message ?? "Failed to load article");
    }
  }

  useEffect(() => {
    if (!episodeId) return;
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [episodeId]);

  useEffect(() => {
    if (!episode) return;
    const inFlight = ["ANALYZING", "PLANNING", "GENERATING", "VERIFYING", "REVISING"].includes(
      episode.status
    );
    if (!inFlight) return;
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [episode?.status]);

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

  if (!episode) {
    return (
      <main className="mx-auto max-w-3xl px-6 py-16">
        <p className="text-sm text-neutral-500">Loading episode…</p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-3xl px-6 py-16">
      <p className="text-sm text-neutral-500">Dev: Article Review — not the real admin UI</p>
      <h1 className="mt-1 text-2xl font-semibold">{episode.title ?? episode.id}</h1>

      <dl className="mt-4 space-y-1 text-sm">
        <div className="flex gap-2">
          <dt className="text-neutral-500">Episode status</dt>
          <dd className="font-mono">{episode.status}</dd>
        </div>
        {episode.last_error && (
          <div className="flex gap-2">
            <dt className="text-neutral-500">Error</dt>
            <dd className="text-red-600">{episode.last_error}</dd>
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
