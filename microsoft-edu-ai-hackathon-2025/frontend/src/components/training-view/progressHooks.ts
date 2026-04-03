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
