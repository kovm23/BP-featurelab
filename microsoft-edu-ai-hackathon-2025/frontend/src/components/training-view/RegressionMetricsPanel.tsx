import type { PredictionMetrics } from "@/lib/api";
import type { TrainingTranslations } from "./translations";
import { MetricStat } from "./metricStats";

export function RegressionMetricsPanel({
  deluxe,
  predictionMetrics,
  tr,
}: {
  deluxe: boolean;
  predictionMetrics: PredictionMetrics;
  tr: TrainingTranslations;
}) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-x-5 gap-y-4">
      <MetricStat deluxe={deluxe} value={(predictionMetrics.mse ?? 0).toFixed(4)} label={tr.mseLowerBetter} tooltip={tr.mseTooltip} />
      <MetricStat deluxe={deluxe} value={(predictionMetrics.mae ?? 0).toFixed(4)} label={tr.maeLowerBetter} tooltip={tr.maeTooltip} />
      <MetricStat
        deluxe={deluxe}
        value={predictionMetrics.correlation != null ? predictionMetrics.correlation.toFixed(4) : "N/A"}
        label={tr.correlationHigherBetter}
        tooltip={tr.correlationTooltip}
        valueClass={
          predictionMetrics.correlation != null && predictionMetrics.correlation > 0.5
            ? "text-green-600"
            : predictionMetrics.correlation != null && predictionMetrics.correlation > 0
              ? "text-yellow-600"
              : "text-red-500"
        }
      />
    </div>
  );
}
