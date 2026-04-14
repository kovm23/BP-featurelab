import { useCallback, useEffect, useState } from "react";
import { fetchJson, HEALTH_URL, QUEUE_INFO_URL, sessionHeaders } from "@/lib/api";

const REQUEST_TIMEOUT_MS = 4000;
const QUEUE_POLL_MS = 5000;

async function fetchWithTimeout<T>(
  input: RequestInfo | URL,
  init?: RequestInit,
  timeoutMs = REQUEST_TIMEOUT_MS
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetchJson<T>(input, {
      ...init,
      signal: controller.signal,
    });
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export function usePipelineRuntime(anyBusy: boolean) {
  const [ollamaOk, setOllamaOk] = useState<boolean | null>(null);
  const [queueBusy, setQueueBusy] = useState(false);
  const [queuedCount, setQueuedCount] = useState(0);

  const recheckOllama = useCallback(() => {
    fetchWithTimeout<{ ok: boolean; ollama: boolean }>(HEALTH_URL, {
      cache: "no-store",
    })
      .then((data) => setOllamaOk(data.ollama))
      .catch(() => {
        /* keep current state */
      });
  }, []);

  useEffect(() => {
    let cancelled = false;
    const checkHealth = (retryOnFail: boolean) => {
      fetchWithTimeout<{ ok: boolean; ollama: boolean }>(HEALTH_URL, {
        cache: "no-store",
      })
        .then((data) => {
          if (!cancelled) setOllamaOk(data.ollama);
        })
        .catch(() => {
          if (!cancelled && retryOnFail) {
            setTimeout(() => checkHealth(false), 5000);
          }
        });
    };
    checkHealth(true);
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!anyBusy) {
      setQueueBusy(false);
      setQueuedCount(0);
      return;
    }
    let cancelled = false;
    let pollTimeoutId: number | null = null;

    const poll = async () => {
      try {
        const data = await fetchWithTimeout<{ busy: boolean; queued: number }>(QUEUE_INFO_URL, {
          cache: "no-store",
          headers: sessionHeaders(),
        });
        if (!cancelled) {
          setQueueBusy(data.busy);
          setQueuedCount(data.queued);
        }
      } catch {
        if (!cancelled) {
          setQueueBusy(false);
          setQueuedCount(0);
        }
      } finally {
        if (!cancelled) {
          pollTimeoutId = window.setTimeout(poll, QUEUE_POLL_MS);
        }
      }
    };

    poll();
    return () => {
      cancelled = true;
      if (pollTimeoutId !== null) {
        window.clearTimeout(pollTimeoutId);
      }
    };
  }, [anyBusy]);

  return {
    ollamaOk,
    queueBusy,
    queuedCount,
    recheckOllama,
  };
}
