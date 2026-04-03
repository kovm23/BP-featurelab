import { CheckCircle2, Download, PlayCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import type {
  FeatureSpec,
  PredictionItem,
  PredictionMetrics,
  TrainResult,
} from "@/lib/api";
import { downloadExperimentZip } from "@/lib/pipelineDownloads";
import { cls, isClassificationMetrics } from "./shared";
import type { TrainingTranslations } from "./translations";

function PredictionMetricsPanel({
  deluxe,
  predictionMetrics,
  tr,
}: {
  deluxe: boolean;
  predictionMetrics: PredictionMetrics;
  tr: TrainingTranslations;
}) {
  const isCls = isClassificationMetrics(predictionMetrics);
  const matchPct = predictionMetrics.total_count > 0
    ? Math.round((predictionMetrics.matched_count / predictionMetrics.total_count) * 100)
    : 0;

  return (
    <div className={`p-3 rounded-lg border ${cls(deluxe, "bg-blue-50 border-blue-200", "bg-blue-900/20 border-blue-800/50")}`}>
      <p className={`text-xs font-bold mb-2 ${cls(deluxe, "text-blue-900", "text-blue-300")}`}>
        {tr.evalTitle}
      </p>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {isCls ? (
          <>
            <MetricStat deluxe={deluxe} value={(predictionMetrics.accuracy ?? 0).toFixed(4)} label={tr.accuracy} />
            <MetricStat deluxe={deluxe} value={(predictionMetrics.balanced_accuracy ?? 0).toFixed(4)} label={tr.balancedAccuracy} />
            <MetricStat deluxe={deluxe} value={(predictionMetrics.f1_macro ?? 0).toFixed(4)} label={tr.f1Macro} />
            <MetricStat deluxe={deluxe} value={(predictionMetrics.precision_macro ?? 0).toFixed(4)} label={tr.precisionMacro} />
            <MetricStat
              deluxe={deluxe}
              value={(predictionMetrics.recall_macro ?? 0).toFixed(4)}
              label={tr.recallMacro}
              valueClass={(predictionMetrics.recall_macro ?? 0) > 0.6 ? "text-green-600" : (predictionMetrics.recall_macro ?? 0) > 0.3 ? "text-yellow-600" : "text-red-500"}
            />
            <MetricStat
              deluxe={deluxe}
              value={(predictionMetrics.mcc ?? 0).toFixed(4)}
              label={tr.matthews}
              valueClass={(predictionMetrics.mcc ?? 0) > 0.5 ? "text-green-600" : (predictionMetrics.mcc ?? 0) > 0.2 ? "text-yellow-600" : "text-red-500"}
            />
            <div className="text-center col-span-2 sm:col-span-4">
              <p className={`text-sm ${cls(deluxe, "text-blue-700", "text-blue-400")}`}>
                {tr.paired}: {predictionMetrics.matched_count}/{predictionMetrics.total_count} ({matchPct}%)
              </p>
            </div>
          </>
        ) : (
          <>
            <MetricStat deluxe={deluxe} value={(predictionMetrics.mse ?? 0).toFixed(4)} label={tr.mseLowerBetter} />
            <MetricStat deluxe={deluxe} value={(predictionMetrics.mae ?? 0).toFixed(4)} label={tr.maeLowerBetter} />
            <MetricStat
              deluxe={deluxe}
              value={predictionMetrics.correlation != null ? predictionMetrics.correlation.toFixed(4) : "N/A"}
              label={tr.correlationHigherBetter}
              valueClass={
                predictionMetrics.correlation != null && predictionMetrics.correlation > 0.5
                  ? "text-green-600"
                  : predictionMetrics.correlation != null && predictionMetrics.correlation > 0
                    ? "text-yellow-600"
                    : "text-red-500"
              }
            />
            <MetricStat
              deluxe={deluxe}
              value={`${matchPct}%`}
              label={`${tr.paired} (${predictionMetrics.matched_count}/${predictionMetrics.total_count})`}
              valueClass={matchPct === 100 ? "text-green-600" : matchPct > 50 ? "text-yellow-600" : "text-red-500"}
            />
          </>
        )}
      </div>

      {isCls && (
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
      )}
    </div>
  );
}

function MetricStat({
  deluxe,
  value,
  label,
  valueClass,
}: {
  deluxe: boolean;
  value: string;
  label: string;
  valueClass?: string;
}) {
  return (
    <div className="text-center">
      <p className={`text-lg font-bold ${valueClass ?? cls(deluxe, "text-blue-700", "text-blue-400")}`}>{value}</p>
      <p className={`text-[10px] ${cls(deluxe, "text-blue-600", "text-blue-500")}`}>{label}</p>
    </div>
  );
}

function ConfidenceStat({
  deluxe,
  label,
  value,
  valueClass,
}: {
  deluxe: boolean;
  label: string;
  value: number | null | undefined;
  valueClass?: string;
}) {
  return (
    <div className={`rounded-md p-2 border ${cls(deluxe, "bg-white/70 border-blue-100", "bg-slate-900/40 border-slate-700")}`}>
      <p className="text-[10px] opacity-70">{label}</p>
      <p className={`text-sm font-semibold ${valueClass ?? ""}`}>
        {value != null ? Number(value).toFixed(4) : "—"}
      </p>
    </div>
  );
}

function ConfusionMatrix({
  labels,
  matrix,
}: {
  labels: string[];
  matrix: number[][];
  deluxe: boolean;
}) {
  const maxCell = Math.max(1, ...matrix.flat());
  return (
    <table className="text-xs border-collapse">
      <thead>
        <tr>
          <th className="px-2 py-1"></th>
          {labels.map((label) => (
            <th key={label} className="px-2 py-1 font-mono whitespace-nowrap">
              {label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {matrix.map((row, rowIdx) => (
          <tr key={labels[rowIdx] ?? rowIdx}>
            <th className="px-2 py-1 text-left font-mono whitespace-nowrap">{labels[rowIdx]}</th>
            {row.map((cell, colIdx) => {
              const intensity = Math.max(0.08, Number(cell) / maxCell);
              const isDiagonal = rowIdx === colIdx;
              return (
                <td
                  key={`${rowIdx}-${colIdx}`}
                  className="px-2 py-1 text-center font-semibold"
                  style={{
                    backgroundColor: isDiagonal
                      ? `rgba(34, 197, 94, ${intensity})`
                      : `rgba(59, 130, 246, ${intensity})`,
                  }}
                >
                  {cell}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function escapeCsvCell(value: unknown) {
  const text = String(value ?? "");
  if (/^[=+@\-]/.test(text)) return `"'${text.replace(/"/g, '""')}"`;
  if (text.includes(",") || text.includes('"') || text.includes("\n")) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

function downloadPredictionsCsv(predictions: PredictionItem[], featureSpec: FeatureSpec | null, filename: string) {
  const hasLabelPredictions = predictions.some((p) => p.predicted_label !== undefined);
  const hasActualPredictionValues = predictions.some((p) => p.actual_score !== undefined || p.actual_label !== undefined);
  const actualColName = predictions.some((p) => p.actual_label !== undefined) ? "actual_label" : "actual_score";
  const featureColumns = featureSpec ? Object.keys(featureSpec) : [];

  const headers = [
    "media_name",
    ...(hasLabelPredictions ? ["predicted_label", "confidence"] : ["predicted_score"]),
    ...(hasActualPredictionValues ? [actualColName] : []),
    "rule_applied",
    ...featureColumns,
  ];

  const csv = [
    headers.join(","),
    ...predictions.map((prediction) =>
      [
        escapeCsvCell(prediction.media_name),
        ...(hasLabelPredictions
          ? [escapeCsvCell(prediction.predicted_label), escapeCsvCell(prediction.confidence)]
          : [escapeCsvCell(prediction.predicted_score)]),
        ...(hasActualPredictionValues
          ? [escapeCsvCell(prediction.actual_label ?? (prediction.actual_score !== undefined ? prediction.actual_score : ""))]
          : []),
        escapeCsvCell(prediction.rule_applied),
        ...featureColumns.map((feature) => escapeCsvCell(prediction.extracted_features[feature])),
      ].join(",")
    ),
  ].join("\n");

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function PredictionResults({
  deluxe,
  tr,
  predictions,
  predictionMetrics,
  featureSpec,
  trainResult,
  testingDataX,
  isPredicting,
  showFeatureCols,
  onToggleFeatureCols,
  onRerunPrediction,
}: {
  deluxe: boolean;
  tr: TrainingTranslations;
  predictions: PredictionItem[];
  predictionMetrics: PredictionMetrics | null;
  featureSpec: FeatureSpec | null;
  trainResult: TrainResult | null;
  testingDataX: Record<string, unknown>[] | null;
  isPredicting: boolean;
  showFeatureCols: boolean;
  onToggleFeatureCols: () => void;
  onRerunPrediction: () => void;
}) {
  const hasLabelPredictions = predictions.some((p) => p.predicted_label !== undefined);
  const hasActualPredictionValues = predictions.some((p) => p.actual_score !== undefined || p.actual_label !== undefined);
  const classificationRowsSummary = hasLabelPredictions
    ? predictions.reduce(
        (acc, pred) => {
          if (pred.actual_label == null) {
            acc.unlabeled += 1;
          } else if (pred.predicted_label === pred.actual_label) {
            acc.correct += 1;
          } else {
            acc.incorrect += 1;
          }
          return acc;
        },
        { correct: 0, incorrect: 0, unlabeled: 0 }
      )
    : null;

  return (
    <div className={`rounded-xl p-4 space-y-3 border ${cls(deluxe, "bg-green-50 border-green-100", "bg-green-900/20 border-green-800/50")}`}>
      <div className={`flex items-center gap-2 ${cls(deluxe, "text-green-800", "text-green-400")}`}>
        <CheckCircle2 className="h-4 w-4" />
        <p className="text-sm font-bold">
          {tr.predictionDone} ({predictions.length} {tr.objects})
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <span className={`text-[11px] px-2 py-1 rounded-full border ${cls(deluxe, "bg-white border-slate-200 text-slate-600", "bg-slate-900/40 border-slate-700 text-slate-300")}`}>
          {tr.predictionSource}: {isClassificationMetrics(predictionMetrics) ? tr.predictionSourceClassification : tr.predictionSourceRegression}
        </span>
      </div>

      {predictionMetrics && (
        <PredictionMetricsPanel deluxe={deluxe} predictionMetrics={predictionMetrics} tr={tr} />
      )}

      {classificationRowsSummary && (
        <div className="flex flex-wrap gap-2">
          <span className={`text-[11px] px-2 py-1 rounded-full border ${cls(deluxe, "bg-white border-slate-200 text-slate-600", "bg-slate-900/40 border-slate-700 text-slate-300")}`}>
            {tr.predictionTableSummary}
          </span>
          <span className={`text-[11px] px-2 py-1 rounded-full border ${cls(deluxe, "bg-emerald-50 border-emerald-200 text-emerald-700", "bg-emerald-900/30 border-emerald-700/60 text-emerald-300")}`}>
            {tr.correctRows}: {classificationRowsSummary.correct}
          </span>
          <span className={`text-[11px] px-2 py-1 rounded-full border ${cls(deluxe, "bg-rose-50 border-rose-200 text-rose-700", "bg-rose-900/30 border-rose-700/60 text-rose-300")}`}>
            {tr.wrongRows}: {classificationRowsSummary.incorrect}
          </span>
          {classificationRowsSummary.unlabeled > 0 && (
            <span className={`text-[11px] px-2 py-1 rounded-full border ${cls(deluxe, "bg-amber-50 border-amber-200 text-amber-700", "bg-amber-900/30 border-amber-700/60 text-amber-300")}`}>
              {tr.unlabeledRows}: {classificationRowsSummary.unlabeled}
            </span>
          )}
        </div>
      )}

      {featureSpec && Object.keys(featureSpec).length > 0 && (
        <button
          onClick={onToggleFeatureCols}
          className={`text-xs underline opacity-60 hover:opacity-100 ${cls(deluxe, "text-slate-500", "text-slate-400")}`}
        >
          {showFeatureCols ? tr.hideFeatureColumns : `${tr.showFeatureColumns} (${Object.keys(featureSpec).length})`}
        </button>
      )}

      <div className={`rounded-lg border overflow-hidden ${cls(deluxe, "border-slate-200", "border-slate-700")}`}>
        <div className="overflow-x-auto max-h-80">
          <table className="w-full text-xs">
            <thead>
              <tr className={cls(deluxe, "bg-slate-50", "bg-slate-900")}>
                <th className="px-2 py-1 text-left font-mono">{tr.mediaName}</th>
                {hasLabelPredictions ? (
                  <>
                    <th className="px-2 py-1 text-left font-mono">{tr.rowStatus}</th>
                    <th className="px-2 py-1 text-left font-mono">{tr.predictedLabel}</th>
                    <th className="px-2 py-1 text-left font-mono">{tr.confidenceLabel}</th>
                  </>
                ) : (
                  <th className="px-2 py-1 text-left font-mono">{tr.predictedScore}</th>
                )}
                {hasActualPredictionValues && (
                  <th className="px-2 py-1 text-left font-mono">
                    {predictions.some((p) => p.actual_label !== undefined) ? tr.actualLabel : tr.actualScore}
                  </th>
                )}
                <th className="px-2 py-1 text-left font-mono">{tr.ruleApplied}</th>
                {showFeatureCols && featureSpec && Object.keys(featureSpec).map((feature) => (
                  <th key={feature} className="px-2 py-1 text-left font-mono">
                    {feature}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {predictions.map((prediction, index) => {
                const hasActualLabel = prediction.actual_label != null;
                const isCorrect = hasActualLabel && prediction.predicted_label === prediction.actual_label;
                const isWrong = hasActualLabel && prediction.predicted_label !== prediction.actual_label;
                const baseRow = index % 2 === 0 ? cls(deluxe, "bg-white", "bg-slate-800/50") : cls(deluxe, "bg-slate-50/50", "bg-slate-900/50");
                const rowTone = !hasLabelPredictions
                  ? baseRow
                  : isCorrect
                    ? cls(deluxe, "bg-emerald-50/70", "bg-emerald-950/30")
                    : isWrong
                      ? cls(deluxe, "bg-rose-50/70", "bg-rose-950/30")
                      : cls(deluxe, "bg-amber-50/60", "bg-amber-950/20");
                const statusPill = isCorrect
                  ? cls(deluxe, "bg-emerald-100 text-emerald-700", "bg-emerald-900/40 text-emerald-300")
                  : isWrong
                    ? cls(deluxe, "bg-rose-100 text-rose-700", "bg-rose-900/40 text-rose-300")
                    : cls(deluxe, "bg-amber-100 text-amber-700", "bg-amber-900/40 text-amber-300");

                return (
                  <tr key={index} className={rowTone}>
                    <td className="px-2 py-1 whitespace-nowrap font-medium">{prediction.media_name}</td>
                    {hasLabelPredictions ? (
                      <>
                        <td className="px-2 py-1 whitespace-nowrap">
                          <span className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold ${statusPill}`}>
                            {isCorrect ? tr.rowCorrect : isWrong ? tr.rowWrong : tr.rowUnlabeled}
                          </span>
                        </td>
                        <td className="px-2 py-1 whitespace-nowrap font-bold text-blue-600">
                          {prediction.predicted_label ?? "—"}
                        </td>
                        <td className="px-2 py-1 whitespace-nowrap font-bold text-indigo-600">
                          {prediction.confidence != null ? Number(prediction.confidence).toFixed(4) : "—"}
                        </td>
                      </>
                    ) : (
                      <td className="px-2 py-1 whitespace-nowrap font-bold text-blue-600">
                        {prediction.predicted_score}
                      </td>
                    )}
                    {hasActualPredictionValues && (
                      <td className={`px-2 py-1 whitespace-nowrap font-bold ${hasLabelPredictions ? (isCorrect ? "text-emerald-600" : isWrong ? "text-rose-600" : "text-amber-600") : "text-green-600"}`}>
                        {prediction.actual_label ?? (prediction.actual_score !== undefined ? prediction.actual_score : "—")}
                      </td>
                    )}
                    <td className="px-2 py-1 whitespace-nowrap font-mono text-[10px]">{prediction.rule_applied}</td>
                    {showFeatureCols && featureSpec && Object.keys(featureSpec).map((feature) => (
                      <td key={feature} className="px-2 py-1 whitespace-nowrap">
                        {String(prediction.extracted_features[feature] ?? "")}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="flex justify-center gap-2 flex-wrap">
        <button
          onClick={() => downloadPredictionsCsv(predictions, featureSpec, `predictions_${new Date().toISOString().slice(0, 10)}.csv`)}
          className="px-3 py-1.5 bg-slate-500 text-white rounded text-xs hover:bg-slate-600 flex items-center gap-1"
        >
          <Download className="w-3 h-3" /> {tr.downloadPredCsv}
        </button>
        <button
          onClick={() =>
            downloadExperimentZip({
              featureSpec: featureSpec ?? undefined,
              trainingDataX: trainResult?.training_data_X ?? undefined,
              testingDataX: testingDataX ?? undefined,
              rules: trainResult?.rules ?? undefined,
              mse: trainResult?.mse ?? undefined,
              predictions: predictions as unknown as Record<string, unknown>[],
              metrics: (predictionMetrics as unknown as Record<string, unknown>) ?? undefined,
            })
          }
          className="px-3 py-1.5 bg-slate-700 text-white rounded text-xs hover:bg-slate-800 flex items-center gap-1"
        >
          <Download className="w-3 h-3" /> {tr.downloadExperiment}
        </button>
      </div>

      {!isPredicting && (
        <div className="flex justify-center mt-2">
          <Button variant="outline" size="sm" onClick={onRerunPrediction}>
            <PlayCircle className="mr-2 h-3 w-3" /> {tr.rerunPrediction}
          </Button>
        </div>
      )}
    </div>
  );
}
