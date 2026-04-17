import React from "react";
import { CheckCircle2, ChevronRight, Download, Loader2, PlayCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { FeatureSpec } from "@/lib/api";
import { downloadTrainingDataCSV } from "@/lib/pipelineDownloads";
import { cls, DatasetTable, FeatureSpecBox, FileDropZone, ProgressBar } from "./shared";
import { OllamaWarning } from "./OllamaWarning";
import type { TrainingTranslations } from "./translations";

export function TrainingExtractionPhasePanel(props: {
  deluxe: boolean;
  uiLanguage: "cs" | "en";
  tr: TrainingTranslations;
  featureSpec: FeatureSpec | null;
  targetVariable: string;
  setFeatureSpec: (spec: FeatureSpec) => void;
  trainZipFile: File | null;
  setTrainZipFile: (file: File | null) => void;
  useExtractionLabels: boolean;
  setUseExtractionLabels: React.Dispatch<React.SetStateAction<boolean>>;
  extractionLabels: File | null;
  setExtractionLabels: React.Dispatch<React.SetStateAction<File | null>>;
  onExtractTraining: (file: File, labelsFile?: File | null) => void;
  isExtracting: boolean;
  trainingDataX: Record<string, unknown>[] | null;
  ollamaOk?: boolean | null;
  recheckOllama?: () => void;
  progress: number;
  progressLabel: string;
  etaText?: string | null;
  extractStalled: boolean;
  onCancel?: () => void;
  onGoToStep?: (step: number) => void;
}) {
  const {
    deluxe,
    uiLanguage,
    tr,
    featureSpec,
    targetVariable,
    setFeatureSpec,
    trainZipFile,
    setTrainZipFile,
    useExtractionLabels,
    setUseExtractionLabels,
    extractionLabels,
    setExtractionLabels,
    onExtractTraining,
    isExtracting,
    trainingDataX,
    ollamaOk,
    recheckOllama,
    progress,
    progressLabel,
    etaText,
    extractStalled,
    onCancel,
    onGoToStep,
  } = props;

  return (
    <div className="space-y-4">
      {featureSpec && (
        <FeatureSpecBox
          deluxe={deluxe}
          uiLanguage={uiLanguage}
          featureSpec={featureSpec}
          targetVariable={targetVariable}
          editable
          onUpdate={setFeatureSpec}
        />
      )}

      <FileDropZone
        deluxe={deluxe}
        uiLanguage={uiLanguage}
        file={trainZipFile}
        onFile={setTrainZipFile}
        accept=".zip"
        inputId="train-zip-upload"
        label={tr.uploadTrainingZip}
        pickLabel={tr.pickTrainingZip}
        selectedLabel={tr.selected}
      />

      {(
        <div className={`p-3 rounded-lg border ${cls(deluxe, "bg-amber-50/50 border-amber-200", "bg-amber-900/20 border-amber-800/50")}`}>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useExtractionLabels}
              onChange={(e) => {
                setUseExtractionLabels(e.target.checked);
                if (!e.target.checked) setExtractionLabels(null);
              }}
              className="rounded"
            />
            <span className={`text-sm font-medium ${cls(deluxe, "text-amber-800", "text-amber-300")}`}>
              {tr.uploadLabelsSeparately}
            </span>
          </label>
          <p className={`text-xs mt-1 ${cls(deluxe, "text-amber-600", "text-amber-400/70")}`}>{tr.labelsAutoloadHint}</p>
          {useExtractionLabels && (
            <div className="mt-2">
              <input
                type="file"
                accept=".csv"
                onChange={(e) => {
                  if (e.target.files?.[0]) setExtractionLabels(e.target.files[0]);
                }}
                className={`text-xs ${cls(deluxe, "text-slate-600", "text-slate-400")}`}
              />
              {extractionLabels && <p className="mt-1 text-xs text-green-500 font-medium">{tr.selectedCsv}: {extractionLabels.name}</p>}
            </div>
          )}
        </div>
      )}

      <div className={`p-3 rounded-lg border ${cls(deluxe, "bg-blue-50/70 border-blue-200", "bg-blue-900/20 border-blue-800/50")}`}>
        <p className={`text-xs ${cls(deluxe, "text-blue-800", "text-blue-200")}`}>
          {tr.labelsMatchingHint}
        </p>
        <p className={`text-xs mt-1 font-mono ${cls(deluxe, "text-blue-700", "text-blue-300/80")}`}>
          {tr.csvFormatHint}
        </p>
      </div>

      {ollamaOk === false && <OllamaWarning deluxe={deluxe} tr={tr} recheckOllama={recheckOllama} />}

      <div className="flex justify-center mt-6">
        <Button
          onClick={() => {
            if (trainZipFile) {
              onExtractTraining(trainZipFile, useExtractionLabels ? extractionLabels : null);
            }
          }}
          disabled={!trainZipFile || isExtracting}
          title={!trainZipFile ? tr.uploadTrainingFirst : undefined}
        >
          {isExtracting ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> {tr.extractingFeatures}
            </>
          ) : (
            <>
              <PlayCircle className="mr-2 h-4 w-4" /> {tr.startExtraction}
            </>
          )}
        </Button>
      </div>

      {isExtracting && (
        <div className="space-y-2">
          <ProgressBar deluxe={deluxe} progress={progress} label={progressLabel} etaText={etaText} />
          {extractStalled && (
            <div className={`flex items-start gap-2 p-2 rounded-lg border text-xs ${cls(deluxe, "bg-amber-50 border-amber-200 text-amber-800", "bg-amber-900/20 border-amber-700/50 text-amber-300")}`}>
              <span aria-hidden="true">⏳</span>
              <span>{tr.extractionStallWarning}</span>
            </div>
          )}
          {onCancel && (
            <Button variant="outline" size="sm" onClick={onCancel} className="text-xs">
              ✕ {tr.stop}
            </Button>
          )}
        </div>
      )}

      {trainingDataX && !isExtracting && (
        <div className={`flex items-center gap-2 p-2 rounded-lg ${cls(deluxe, "bg-green-50 text-green-700", "bg-green-900/30 text-green-400")}`}>
          <CheckCircle2 className="h-4 w-4" />
          <span className="text-sm font-medium">
            {tr.extractionDone} — {trainingDataX.length} {tr.rows}
          </span>
        </div>
      )}

      {trainingDataX && (
        <>
          <DatasetTable deluxe={deluxe} data={trainingDataX} title={tr.trainingDatasetTitle} />
          <div className="flex justify-center mt-2">
            <button
              onClick={() => downloadTrainingDataCSV(trainingDataX)}
              className="px-3 py-1.5 bg-slate-600 text-white rounded text-xs hover:bg-slate-700 flex items-center gap-1"
            >
              <Download className="w-3 h-3" /> {tr.downloadTrainingX}
            </button>
          </div>
          <div className="flex justify-center mt-4">
            <Button onClick={() => onGoToStep?.(3)}>
              <ChevronRight className="mr-2 h-4 w-4" /> {tr.continue3}
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
