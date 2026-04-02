import { useEffect, useRef, useState } from "react";

export function useElapsedTimer(active: boolean) {
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    if (!active) {
      setSecs(0);
      return;
    }
    setSecs(0);
    const id = setInterval(() => setSecs((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [active]);
  return secs;
}

export function formatDurationShort(totalSeconds: number) {
  const seconds = Math.max(0, Math.round(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

export function useEstimatedRemaining(progress: number, active: boolean) {
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);
  const samplesRef = useRef<Array<{ time: number; progress: number }>>([]);
  const lastProgressRef = useRef<number | null>(null);

  const reset = () => {
    samplesRef.current = [];
    lastProgressRef.current = null;
    setEtaSeconds(null);
  };

  useEffect(() => {
    if (!active) {
      reset();
    } else {
      const now = Date.now();
      samplesRef.current = [{ time: now, progress }];
      lastProgressRef.current = progress;
    }
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!active) return;
    if (progress < 8 || progress >= 99) {
      if (progress >= 99) setEtaSeconds(null);
      lastProgressRef.current = progress;
      return;
    }

    const now = Date.now();
    const lastProgress = lastProgressRef.current;

    if (lastProgress !== null && progress < lastProgress) {
      samplesRef.current = [{ time: now, progress }];
      setEtaSeconds(null);
      lastProgressRef.current = progress;
      return;
    }

    if (lastProgress === progress) return;

    samplesRef.current = [...samplesRef.current, { time: now, progress }]
      .filter((sample) => now - sample.time <= 90_000)
      .slice(-8);
    lastProgressRef.current = progress;

    const first = samplesRef.current[0];
    const last = samplesRef.current[samplesRef.current.length - 1];
    if (!first || !last) return;

    const deltaProgress = last.progress - first.progress;
    const deltaSeconds = (last.time - first.time) / 1000;
    if (deltaProgress <= 0 || deltaSeconds < 5) return;

    const speedPerSecond = deltaProgress / deltaSeconds;
    const rawEta = (100 - progress) / speedPerSecond;
    if (!Number.isFinite(rawEta) || rawEta <= 0) return;

    const rounded = Math.max(1, Math.round(rawEta));
    setEtaSeconds((prev) => (prev == null ? rounded : Math.min(prev, rounded)));
  }, [progress, active]);

  useEffect(() => {
    if (!active) return;
    const intervalId = setInterval(() => {
      setEtaSeconds((prev) => {
        if (prev == null) return null;
        return prev > 1 ? prev - 1 : 1;
      });
    }, 1000);
    return () => clearInterval(intervalId);
  }, [active]);

  return etaSeconds;
}

export function useProgressStall(progress: number, active: boolean, thresholdMs = 90_000) {
  const [stalled, setStalled] = useState(false);
  const lastRef = useRef({ val: -1, time: 0 });

  useEffect(() => {
    if (!active) {
      setStalled(false);
      return;
    }
    if (progress !== lastRef.current.val) {
      lastRef.current = { val: progress, time: Date.now() };
      setStalled(false);
      return;
    }
    const id = setInterval(() => {
      if (Date.now() - lastRef.current.time > thresholdMs) setStalled(true);
    }, 5000);
    return () => clearInterval(id);
  }, [progress, active, thresholdMs]);

  return stalled;
}
