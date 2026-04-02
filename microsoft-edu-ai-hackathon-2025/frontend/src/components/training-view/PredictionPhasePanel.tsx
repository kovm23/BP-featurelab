import React from "react";
import { Loader2, PlayCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { FeatureSpec, PredictionItem, PredictionMetrics, TrainResult } from "@/lib/api";
import { cls, DatasetTable, ProgressBar } from "./shared";
import { PredictionResults } from "./PredictionResults";
import type { TrainingTranslations } from "./translations";

export function PredictionPhasePanel(props: {
  deluxe: boolean;
  tr: TrainingTranslations;
  testingDataX: Record<string, unknown>[] | null;
  showPredictForm: boolean;
  useTestingLabels: boolean;
  setUseTestingLabels: React.Dispatch<React.SetStateAction<boolean>>;
  testingLabels: File | null;
  setTestingLabels: React.Dispatch<React.SetStateAction<File | null>>;
  onPredict: (labelsFile?: File | null) => void;
  isPredicting: boolean;
  predictSecs: number;
  progress: number;
  progressLabel: string;
  etaText?: string | null;
  onCancel?: () => void;
  predictions: PredictionItem[] | null;
  predictionMetrics: PredictionMetrics | null;
  featureSpec: FeatureSpec | null;
  trainResult: TrainResult | null;
  showFeatureCols: boolean;
  setShowFeatureCols: React.Dispatch<React.SetStateAction<boolean>>;
  setShowPredictForm: React.Dispatch<React.SetStateAction<boolean>>;
}) {
  const {
    deluxe,
    tr,
    testingDataX,
    showPredictForm,
    useTestingLabels,
    setUseTestingLabels,
    testingLabels,
    setTestingLabels,
    onPredict,
    isPredicting,
    predictSecs,
    progress,
    progressLabel,
    etaText,
    onCancel,
    predictions,
    predictionMetrics,
    featureSpec,
    trainResult,
    showFeatureCols,
    setShowFeatureCols,
    setShowPredictForm,
  } = props;

  return (
    <div className="space-y-4">
      {testingDataX && <DatasetTable deluxe={deluxe} data={testingDataX} title="Testing Dataset X" />}

      {showPredictForm && (
        <div className={`p-3 rounded-lg border ${cls(deluxe, "bg-amber-50/50 border-amber-200", "bg-amber-900/20 border-amber-800/50")}`}>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useTestingLabels}
              onChange={(e) => {
                setUseTestingLabels(e.target.checked);
                if (!e.target.checked) setTestingLabels(null);
              }}
              className="rounded"
            />
            <span className={`text-sm font-medium ${cls(deluxe, "text-amber-800", "text-amber-300")}`}>
              {tr.uploadTestingY}
            </span>
          </label>
          <p className={`text-xs mt-1 ${cls(deluxe, "text-amber-600", "text-amber-400/70")}`}>
            {tr.uploadTestingYHint}
          </p>
          {useTestingLabels && (
            <div className="mt-2">
              <input
                type="file"
                accept=".csv"
                onChange={(e) => {
                  if (e.target.files?.[0]) setTestingLabels(e.target.files[0]);
                }}
                className={`text-xs ${cls(deluxe, "text-slate-600", "text-slate-400")}`}
              />
              {testingLabels && <p className="mt-1 text-xs text-green-500 font-medium">CSV: {testingLabels.name}</p>}
            </div>
          )}
        </div>
      )}

      {showPredictForm && (
        <div className="flex justify-center mt-6">
          <Button onClick={() => onPredict(useTestingLabels ? testingLabels : null)} disabled={isPredicting}>
            {isPredicting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> {tr.predicting} ({predictSecs}s)
              </>
            ) : (
              <>
                <PlayCircle className="mr-2 h-4 w-4" /> {tr.startPrediction}
              </>
            )}
          </Button>
        </div>
      )}

      {isPredicting && (
        <div className="space-y-2 mt-4">
          <ProgressBar deluxe={deluxe} progress={progress} label={progressLabel || tr.predictionInProgressLabel} etaText={etaText} />
          {onCancel && (
            <div className="flex justify-center">
              <Button variant="outline" size="sm" onClick={onCancel} className="text-xs">
                ✕ {tr.stop}
              </Button>
            </div>
          )}
        </div>
      )}

      {predictions && predictions.length > 0 && (
        <PredictionResults
          deluxe={deluxe}
          tr={tr}
          predictions={predictions}
          predictionMetrics={predictionMetrics}
          featureSpec={featureSpec}
          trainResult={trainResult}
          testingDataX={testingDataX}
          isPredicting={isPredicting}
          showFeatureCols={showFeatureCols}
          onToggleFeatureCols={() => setShowFeatureCols((value) => !value)}
          onRerunPrediction={() => setShowPredictForm(true)}
        />
      )}
    </div>
  );
}
