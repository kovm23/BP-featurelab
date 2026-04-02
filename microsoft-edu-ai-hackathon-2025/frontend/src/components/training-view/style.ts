import type { PredictionMetrics } from "@/lib/api";

export function cls(deluxe: boolean, light: string, dark: string) {
  return deluxe ? dark : light;
}

export function isClassificationMetrics(metrics: PredictionMetrics | null): boolean {
  if (!metrics) return false;
  return metrics.mode === "classification" || typeof metrics.accuracy === "number";
}
