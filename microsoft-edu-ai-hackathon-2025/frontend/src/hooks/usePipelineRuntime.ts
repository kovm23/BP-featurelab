import { useCallback, useEffect, useState } from "react";
import { fetchJson, HEALTH_URL, QUEUE_INFO_URL, sessionHeaders } from "@/lib/api";

export function usePipelineRuntime(anyBusy: boolean) {
  const [ollamaOk, setOllamaOk] = useState<boolean | null>(null);
  const [queueBusy, setQueueBusy] = useState(false);
  const [queuedCount, setQueuedCount] = useState(0);

  const recheckOllama = useCallback(() => {
    fetchJson<{ ok: boolean; ollama: boolean }>(HEALTH_URL, { cache: "no-store" })
      .then((data) => setOllamaOk(data.ollama))
      .catch(() => {
        /* keep current state */
      });
  }, []);

  useEffect(() => {
    let cancelled = false;
    const checkHealth = (retryOnFail: boolean) => {
      fetchJson<{ ok: boolean; ollama: boolean }>(HEALTH_URL, { cache: "no-store" })
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
    const poll = () => {
      fetchJson<{ busy: boolean; queued: number }>(QUEUE_INFO_URL, {
        cache: "no-store",
        headers: sessionHeaders(),
      })
        .then((data) => {
          if (!cancelled) {
            setQueueBusy(data.busy);
            setQueuedCount(data.queued);
          }
        })
        .catch(() => {
          if (!cancelled) {
            setQueueBusy(false);
            setQueuedCount(0);
          }
        });
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [anyBusy]);

  return {
    ollamaOk,
    queueBusy,
    queuedCount,
    recheckOllama,
  };
}
