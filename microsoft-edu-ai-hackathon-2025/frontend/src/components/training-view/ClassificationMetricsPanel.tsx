import type { PredictionMetrics } from "@/lib/api";
import { cls } from "./shared";
import type { TrainingTranslations } from "./translations";
import { ConfidenceStat, ConfusionMatrix, MetricStat } from "./metricStats";

export function ClassificationMetricsPanel({
  deluxe,
  predictionMetrics,
  tr,
}: {
  deluxe: boolean;
  predictionMetrics: PredictionMetrics;
  tr: TrainingTranslations;
}) {
  return (
    <>
      <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-x-5 gap-y-4">
        <MetricStat deluxe={deluxe} value={(predictionMetrics.accuracy ?? 0).toFixed(4)} label={tr.accuracy} tooltip={tr.accuracyTooltip} />
        <MetricStat deluxe={deluxe} value={(predictionMetrics.balanced_accuracy ?? 0).toFixed(4)} label={tr.balancedAccuracy} tooltip={tr.balancedAccuracyTooltip} />
        <MetricStat deluxe={deluxe} value={(predictionMetrics.f1_macro ?? 0).toFixed(4)} label={tr.f1Macro} tooltip={tr.f1MacroTooltip} />
        <MetricStat deluxe={deluxe} value={(predictionMetrics.precision_macro ?? 0).toFixed(4)} label={tr.precisionMacro} tooltip={tr.precisionMacroTooltip} />
        <MetricStat
          deluxe={deluxe}
          value={(predictionMetrics.recall_macro ?? 0).toFixed(4)}
          label={tr.recallMacro}
          tooltip={tr.recallMacroTooltip}
          valueClass={(predictionMetrics.recall_macro ?? 0) > 0.6 ? "text-green-600" : (predictionMetrics.recall_macro ?? 0) > 0.3 ? "text-yellow-600" : "text-red-500"}
        />
        <MetricStat
          deluxe={deluxe}
          value={(predictionMetrics.mcc ?? 0).toFixed(4)}
          label={tr.matthews}
          tooltip={tr.matthewsTooltip}
          valueClass={(predictionMetrics.mcc ?? 0) > 0.5 ? "text-green-600" : (predictionMetrics.mcc ?? 0) > 0.2 ? "text-yellow-600" : "text-red-500"}
        />
      </div>

      <div className="mt-3 space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <ConfidenceStat deluxe={deluxe} label={tr.avgConfidence} value={predictionMetrics.avg_confidence} />
          <ConfidenceStat deluxe={deluxe} label={tr.correctConfidence} value={predictionMetrics.correct_confidence_avg} valueClass="text-green-600" />
          <ConfidenceStat deluxe={deluxe} label={tr.wrongConfidence} value={predictionMetrics.incorrect_confidence_avg} valueClass="text-amber-600" />
        </div>

        {predictionMetrics.class_metrics && predictionMetrics.class_metrics.length > 0 && (
          <details className={`rounded-md border p-2 ${cls(deluxe, "bg-white/70 border-blue-100", "bg-slate-900/40 border-slate-700")}`}>
            <summary className="cursor-pointer text-xs font-semibold">{tr.classBreakdown}</summary>
            <div className="mt-2 overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className={cls(deluxe, "text-slate-600", "text-slate-400")}>
                    <th className="px-2 py-1 text-left">{tr.label}</th>
                    <th className="px-2 py-1 text-left">{tr.precision}</th>
                    <th className="px-2 py-1 text-left">{tr.recall}</th>
                    <th className="px-2 py-1 text-left">F1</th>
                    <th className="px-2 py-1 text-left">{tr.support}</th>
                  </tr>
                </thead>
                <tbody>
                  {predictionMetrics.class_metrics.map((metric) => (
                    <tr key={metric.label} className={cls(deluxe, "border-t border-slate-200", "border-t border-slate-700")}>
                      <td className="px-2 py-1 font-medium">{metric.label}</td>
                      <td className="px-2 py-1">{metric.precision.toFixed(4)}</td>
                      <td className="px-2 py-1">{metric.recall.toFixed(4)}</td>
                      <td className="px-2 py-1">{metric.f1.toFixed(4)}</td>
                      <td className="px-2 py-1">{metric.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}

        {predictionMetrics.labels && predictionMetrics.confusion_matrix && predictionMetrics.labels.length > 0 && (
          <details className={`rounded-md border p-2 ${cls(deluxe, "bg-white/70 border-blue-100", "bg-slate-900/40 border-slate-700")}`}>
            <summary className="cursor-pointer text-xs font-semibold">{tr.confusionMatrix}</summary>
            <div className="mt-2 overflow-x-auto">
              <p className={`mb-2 text-[10px] ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>
                {tr.actualAxis} × {tr.predictedAxis}
              </p>
              <ConfusionMatrix deluxe={deluxe} labels={predictionMetrics.labels} matrix={predictionMetrics.confusion_matrix} />
            </div>
          </details>
        )}
      </div>
    </>
  );
}
