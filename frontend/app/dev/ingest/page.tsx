"use client";

import { useEffect, useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Episode = {
  id: string;
  youtube_video_id: string;
  status: string;
  title: string | null;
  last_error: string | null;
};

export default function DevIngestPage() {
  const [url, setUrl] = useState("");
  const [episode, setEpisode] = useState<Episode | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleIngest(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/episodes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ youtube_url: url }),
      });
      const body = await response.json();
      if (!response.ok) {
        setError(body.message ?? "Request failed");
        setEpisode(null);
      } else {
        setEpisode(body);
      }
    } catch {
      setError("Could not reach the backend API");
    } finally {
      setLoading(false);
    }
  }

  // Poll status while a job is in flight -- this is a dev tool, not the
  // real processing UI (that's Phase 6), so a simple interval is enough.
  useEffect(() => {
    if (!episode || episode.status !== "INGESTING") return;
    const interval = setInterval(async () => {
      const response = await fetch(`${API_BASE_URL}/api/v1/episodes/${episode.id}`);
      if (response.ok) setEpisode(await response.json());
    }, 2000);
    return () => clearInterval(interval);
  }, [episode]);

  return (
    <main className="mx-auto max-w-md px-6 py-16">
      <h1 className="text-xl font-semibold">Dev: Ingest Episode</h1>
      <p className="mt-1 text-sm text-neutral-500">
        Internal tool for Phase 2 — not the product admin dashboard.
      </p>

      <form onSubmit={handleIngest} className="mt-6 flex gap-2">
        <input
          type="url"
          required
          placeholder="https://www.youtube.com/watch?v=..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="flex-1 rounded border border-neutral-300 px-3 py-2 text-sm"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded bg-neutral-900 px-4 py-2 text-sm text-white disabled:opacity-50"
        >
          {loading ? "..." : "Ingest"}
        </button>
      </form>

      {error && <p className="mt-4 text-sm text-red-600">{error}</p>}

      {episode && (
        <dl className="mt-6 space-y-1 text-sm">
          <div className="flex justify-between">
            <dt className="text-neutral-500">Episode ID</dt>
            <dd className="font-mono">{episode.id}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-neutral-500">Status</dt>
            <dd className="font-mono">{episode.status}</dd>
          </div>
          {episode.title && (
            <div className="flex justify-between">
              <dt className="text-neutral-500">Title</dt>
              <dd>{episode.title}</dd>
            </div>
          )}
          {episode.last_error && (
            <div className="flex justify-between">
              <dt className="text-neutral-500">Error</dt>
              <dd className="text-red-600">{episode.last_error}</dd>
            </div>
          )}
        </dl>
      )}
    </main>
  );
}
