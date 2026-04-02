import { Loader2, PlayCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { FeatureSpec, TrainResult } from "@/lib/api";
import { cls, DatasetTable, FeatureSpecBox, ProgressBar } from "./shared";
import { TrainingResultsCard } from "./TrainingResultsCard";
import type { TrainingTranslations } from "./translations";

export function TrainingPhasePanel(props: {
  deluxe: boolean;
  uiLanguage: "cs" | "en";
  tr: TrainingTranslations;
  featureSpec: FeatureSpec | null;
  targetVariable: string;
  trainingDataX: Record<string, unknown>[] | null;
  datasetYColumns: string[] | null;
  targetColumn: string;
  setTargetColumn: (value: string) => void;
  onTrain: (targetColumn: string) => void;
  isTraining: boolean;
  trainSecs: number;
  trainResult: TrainResult | null;
  progress: number;
  progressLabel: string;
  etaText?: string | null;
  onCancel?: () => void;
  onGoToStep?: (step: number) => void;
}) {
  const {
    deluxe,
    uiLanguage,
    tr,
    featureSpec,
    targetVariable,
    trainingDataX,
    datasetYColumns,
    targetColumn,
    setTargetColumn,
    onTrain,
    isTraining,
    trainSecs,
    trainResult,
    progress,
    progressLabel,
    etaText,
    onCancel,
    onGoToStep,
  } = props;

  return (
    <div className="space-y-4">
      {featureSpec && (
        <FeatureSpecBox deluxe={deluxe} uiLanguage={uiLanguage} featureSpec={featureSpec} targetVariable={targetVariable} />
      )}

      {trainingDataX && <DatasetTable deluxe={deluxe} data={trainingDataX} title="Training Dataset X" />}

      <div>
        <label className={`block text-sm font-medium mb-1 ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
          {tr.targetColumnLabel}
        </label>
        {datasetYColumns && datasetYColumns.length > 0 ? (
          <select
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            className={`w-full p-2 text-sm rounded-lg border outline-none ${cls(
              deluxe,
              "bg-slate-50 border-slate-200 text-slate-900",
              "bg-slate-900 border-slate-700 text-white"
            )}`}
          >
            <option value="">{tr.selectColumn}</option>
            {datasetYColumns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            className={`w-full p-2 text-sm rounded-lg border outline-none ${cls(
              deluxe,
              "bg-slate-50 border-slate-200 text-slate-900",
              "bg-slate-900 border-slate-700 text-white"
            )}`}
            placeholder={tr.targetPlaceholder}
          />
        )}
      </div>

      <div className="flex justify-center mt-6">
        <Button onClick={() => onTrain(targetColumn)} disabled={!targetColumn || isTraining} title={!targetColumn ? tr.pickTargetColumn : undefined}>
          {isTraining ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> {tr.trainingInProgressLabel} ({trainSecs}s)
            </>
          ) : (
            <>
              <PlayCircle className="mr-2 h-4 w-4" /> {tr.startTraining}
            </>
          )}
        </Button>
      </div>

      {isTraining && (
        <div className="space-y-2 mt-4">
          <ProgressBar deluxe={deluxe} progress={progress} label={progressLabel || tr.trainingInProgressLabel} etaText={etaText} />
          {onCancel && (
            <div className="flex justify-center">
              <Button variant="outline" size="sm" onClick={onCancel} className="text-xs">
                ✕ {tr.stop}
              </Button>
            </div>
          )}
        </div>
      )}

      {trainResult && <TrainingResultsCard deluxe={deluxe} trainResult={trainResult} tr={tr} onGoToStep={onGoToStep} />}
    </div>
  );
}
