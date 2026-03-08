import { STATUS_URL } from "@/lib/api";
import type { StatusPayload } from "@/lib/api";

export async function pollProgress(
  jobId: string,
  onTick: (s: StatusPayload) => void,
  signal: AbortSignal
) {
  while (!signal.aborted) {
    try {
      const r = await fetch(STATUS_URL(jobId), {
        signal,
        cache: "no-store",
      });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const s: StatusPayload = await r.json();
      onTick(s);
      if (s.done || s.progress >= 100) break;
    } catch {
      /* silent backoff */
    }
    await new Promise((res) => setTimeout(res, 600));
  }
}
