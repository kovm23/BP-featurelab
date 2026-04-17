import { fetchJson, STATUS_URL, sessionHeaders } from "@/lib/api";
import type { StatusPayload } from "@/lib/api";

const _POLL_MIN_MS = 600;
const _POLL_MAX_MS = 5000;

export type PollReason = "done" | "aborted" | "lost";

export interface PollOptions {
  onTick: (s: StatusPayload) => void;
  onReconnecting?: (attempts: number) => void;
  signal: AbortSignal;
}

export async function pollProgress(
  jobId: string,
  onTick: (s: StatusPayload) => void,
  signal: AbortSignal,
  onReconnecting?: (attempts: number) => void,
): Promise<PollReason> {
  let delay = _POLL_MIN_MS;
  let reconnectAttempts = 0;
  while (!signal.aborted) {
    try {
      const s = await fetchJson<StatusPayload>(STATUS_URL(jobId), {
        signal,
        cache: "no-store",
        headers: sessionHeaders(),
      });
      // Job disappeared from registry (TTL cleanup, worker restart, etc.)
      if (s.error && s.done === undefined) {
        return "lost";
      }
      if (reconnectAttempts > 0) {
        reconnectAttempts = 0;
        onReconnecting?.(0);
      }
      onTick(s);
      if (s.done || s.progress >= 100) return "done";
      delay = _POLL_MIN_MS;
    } catch {
      if (signal.aborted) break;
      reconnectAttempts += 1;
      onReconnecting?.(reconnectAttempts);
      delay = Math.min(delay * 2, _POLL_MAX_MS);
    }
    await new Promise((res) => setTimeout(res, delay));
  }
  return "aborted";
}
