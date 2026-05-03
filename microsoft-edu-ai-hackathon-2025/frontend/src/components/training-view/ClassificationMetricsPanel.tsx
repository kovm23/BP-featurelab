import { useState } from "react";
import { Download, Loader2 } from "lucide-react";
import type { PredictionMetrics } from "@/lib/api";
import { cls } from "./shared";
import type { TrainingTranslations } from "./translations";
import { ConfidenceStat, ConfusionMatrix, MetricStat } from "./metricStats";

function exportConfusionMatrixToPng(
  labels: string[],
  matrix: number[][],
): Promise<void> {
  return new Promise((resolve, reject) => {
    const n = labels.length;
    const CELL = 64;
    const LABEL_W = 120;
    const LABEL_H = 80;
    const PAD = 20;

    const width = LABEL_W + n * CELL + PAD;
    const height = LABEL_H + n * CELL + PAD;

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) { reject(new Error("Canvas not supported")); return; }

    // Background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    // Row-normalised values for color
    const rowNorm = matrix.map((row) => {
      const s = row.reduce((a, b) => a + b, 0) || 1;
      return row.map((v) => v / s);
    });

    // Cells
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const norm = rowNorm[i][j];
        // Blue scale: 0 → #eff6ff, 1 → #1d4ed8
        const r = Math.round(239 - norm * (239 - 29));
        const g = Math.round(246 - norm * (246 - 78));
        const b = Math.round(255 - norm * (255 - 216));
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(LABEL_W + j * CELL, LABEL_H + i * CELL, CELL, CELL);

        // Cell border
        ctx.strokeStyle = "#e2e8f0";
        ctx.lineWidth = 1;
        ctx.strokeRect(LABEL_W + j * CELL, LABEL_H + i * CELL, CELL, CELL);

        // Cell value
        ctx.fillStyle = norm > 0.5 ? "#ffffff" : "#1e293b";
        ctx.font = "bold 13px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(
          String(matrix[i][j]),
          LABEL_W + j * CELL + CELL / 2,
          LABEL_H + i * CELL + CELL / 2,
        );
      }
    }

    // Axis labels (predicted = x, actual = y)
    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let j = 0; j < n; j++) {
      // Predicted (top)
      ctx.save();
      ctx.translate(LABEL_W + j * CELL + CELL / 2, LABEL_H - 10);
      ctx.rotate(-Math.PI / 4);
      ctx.textAlign = "left";
      ctx.fillText(labels[j], 0, 0);
      ctx.restore();
    }
    for (let i = 0; i < n; i++) {
      // Actual (left)
      ctx.textAlign = "right";
      ctx.fillText(labels[i], LABEL_W - 6, LABEL_H + i * CELL + CELL / 2);
    }

    // Axis titles
    ctx.font = "bold 12px sans-serif";
    ctx.fillStyle = "#111827";
    ctx.textAlign = "center";
    ctx.fillText("Predicted", LABEL_W + (n * CELL) / 2, height - 6);
    ctx.save();
    ctx.translate(10, LABEL_H + (n * CELL) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Actual", 0, 0);
    ctx.restore();

    canvas.toBlob((blob) => {
      if (!blob) { reject(new Error("Canvas export failed")); return; }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "confusion_matrix.png";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 500);
      resolve();
    }, "image/png");
  });
}

export function ClassificationMetricsPanel({
  deluxe,
  predictionMetrics,
  tr,
}: {
  deluxe: boolean;
  predictionMetrics: PredictionMetrics;
  tr: TrainingTranslations;
}) {
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  function handleDownload() {
    if (!predictionMetrics.labels || !predictionMetrics.confusion_matrix) return;
    setDownloading(true);
    setDownloadError(null);
    exportConfusionMatrixToPng(predictionMetrics.labels, predictionMetrics.confusion_matrix)
      .then(() => setDownloading(false))
      .catch((err: Error) => {
        setDownloading(false);
        setDownloadError(err.message ?? "Export failed");
      });
  }

  return (
    <>
      <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-x-5 gap-y-4">
        <div>
          <MetricStat deluxe={deluxe} value={(predictionMetrics.accuracy ?? 0).toFixed(4)} label={tr.accuracy} tooltip={tr.accuracyTooltip} />
          {predictionMetrics.baseline_accuracy != null && (
            <p className={`text-[10px] mt-0.5 text-center ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>
              baseline: {(predictionMetrics.baseline_accuracy * 100).toFixed(1)}%
              {" "}(
              <span className={(predictionMetrics.accuracy ?? 0) >= predictionMetrics.baseline_accuracy ? "text-green-600" : "text-red-500"}>
                {((predictionMetrics.accuracy ?? 0) - predictionMetrics.baseline_accuracy) >= 0 ? "+" : ""}
                {(((predictionMetrics.accuracy ?? 0) - predictionMetrics.baseline_accuracy) * 100).toFixed(1)}pp
              </span>
              )
            </p>
          )}
        </div>
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
              <ConfusionMatrix labels={predictionMetrics.labels} matrix={predictionMetrics.confusion_matrix} />
              <div className="mt-2 flex items-center justify-end gap-2">
                {downloadError && (
                  <span className="text-xs text-red-500">{downloadError}</span>
                )}
                <button
                  type="button"
                  onClick={handleDownload}
                  disabled={downloading}
                  className={`px-2 py-1 rounded text-xs flex items-center gap-1 disabled:opacity-60 ${cls(deluxe, "bg-slate-500 hover:bg-slate-600 text-white", "bg-slate-600 hover:bg-slate-500 text-white")}`}
                >
                  {downloading
                    ? <Loader2 className="w-3 h-3 animate-spin" />
                    : <Download className="w-3 h-3" />}
                  Download as PNG
                </button>
              </div>
            </div>
          </details>
        )}
      </div>
    </>
  );
}
