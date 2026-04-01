import { fetchJson, STATUS_URL, sessionHeaders } from "@/lib/api";
import type { StatusPayload } from "@/lib/api";

const _POLL_MIN_MS = 600;
const _POLL_MAX_MS = 5000;

export async function pollProgress(
  jobId: string,
  onTick: (s: StatusPayload) => void,
  signal: AbortSignal
) {
  let delay = _POLL_MIN_MS;
  while (!signal.aborted) {
    try {
      const s = await fetchJson<StatusPayload>(STATUS_URL(jobId), {
        signal,
        cache: "no-store",
        headers: sessionHeaders(),
      });
      onTick(s);
      if (s.done || s.progress >= 100) break;
      // Job disappeared from registry (TTL cleanup or server restart)
      if (s.error && s.done === undefined) break;
      // Reset backoff on success
      delay = _POLL_MIN_MS;
    } catch {
      // Increase delay on transient error (network/server not ready)
      delay = Math.min(delay * 2, _POLL_MAX_MS);
    }
    await new Promise((res) => setTimeout(res, delay));
  }
}
