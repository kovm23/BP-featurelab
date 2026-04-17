import { CheckCircle2, ChevronRight, Download, Loader2, PlayCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { FeatureSpec } from "@/lib/api";
import { downloadTrainingDataCSV } from "@/lib/pipelineDownloads";
import { cls, DatasetTable, FeatureSpecBox, FileDropZone, ProgressBar } from "./shared";
import { OllamaWarning } from "./OllamaWarning";
import type { TrainingTranslations } from "./translations";

export function TestingExtractionPhasePanel(props: {
  deluxe: boolean;
  uiLanguage: "cs" | "en";
  tr: TrainingTranslations;
  featureSpec: FeatureSpec | null;
  targetVariable: string;
  testZipFile: File | null;
  setTestZipFile: (file: File | null) => void;
  onExtractTesting: (file: File) => void;
  isExtractingTest: boolean;
  testingDataX: Record<string, unknown>[] | null;
  ollamaOk?: boolean | null;
  recheckOllama?: () => void;
  progress: number;
  progressLabel: string;
  etaText?: string | null;
  testExtractStalled: boolean;
  onCancel?: () => void;
  onGoToStep?: (step: number) => void;
}) {
  const {
    deluxe,
    uiLanguage,
    tr,
    featureSpec,
    targetVariable,
    testZipFile,
    setTestZipFile,
    onExtractTesting,
    isExtractingTest,
    testingDataX,
    ollamaOk,
    recheckOllama,
    progress,
    progressLabel,
    etaText,
    testExtractStalled,
    onCancel,
    onGoToStep,
  } = props;

  return (
    <div className="space-y-4">
      {featureSpec && (
        <FeatureSpecBox deluxe={deluxe} uiLanguage={uiLanguage} featureSpec={featureSpec} targetVariable={targetVariable} />
      )}

      <FileDropZone
        deluxe={deluxe}
        uiLanguage={uiLanguage}
        file={testZipFile}
        onFile={setTestZipFile}
        accept=".zip"
        inputId="test-zip-upload"
        label={tr.uploadTestingZip}
        pickLabel={tr.pickTestingZip}
        selectedLabel={tr.selected}
      />

      <div className={`p-3 rounded-lg border ${cls(deluxe, "bg-blue-50/70 border-blue-200", "bg-blue-900/20 border-blue-800/50")}`}>
        <p className={`text-xs font-mono ${cls(deluxe, "text-blue-700", "text-blue-300/80")}`}>
          {tr.csvFormatHint}
        </p>
      </div>

      {ollamaOk === false && <OllamaWarning deluxe={deluxe} tr={tr} recheckOllama={recheckOllama} />}

      <div className="flex justify-center mt-6">
        <Button
          onClick={() => {
            if (testZipFile) {
              onExtractTesting(testZipFile);
            }
          }}
          disabled={!testZipFile || isExtractingTest}
          title={!testZipFile ? tr.uploadTestingFirst : undefined}
        >
          {isExtractingTest ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> {tr.extractingFeatures}
            </>
          ) : (
            <>
              <PlayCircle className="mr-2 h-4 w-4" /> {tr.startTestExtraction}
            </>
          )}
        </Button>
      </div>

      {isExtractingTest && (
        <div className="space-y-2">
          <ProgressBar deluxe={deluxe} progress={progress} label={progressLabel} etaText={etaText} />
          {testExtractStalled && (
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

      {testingDataX && !isExtractingTest && (
        <div className={`flex items-center gap-2 p-2 rounded-lg ${cls(deluxe, "bg-green-50 text-green-700", "bg-green-900/30 text-green-400")}`}>
          <CheckCircle2 className="h-4 w-4" />
          <span className="text-sm font-medium">
            {tr.testingExtractionDone} — {testingDataX.length} {tr.rows}
          </span>
        </div>
      )}

      {testingDataX && (
        <>
          <DatasetTable deluxe={deluxe} data={testingDataX} title={tr.testingDatasetTitle} />
          <div className="flex justify-center mt-2">
            <button
              onClick={() =>
                downloadTrainingDataCSV(testingDataX, `testing_dataset_X_${new Date().toISOString().slice(0, 10)}.csv`)
              }
              className="px-3 py-1.5 bg-slate-600 text-white rounded text-xs hover:bg-slate-700 flex items-center gap-1"
            >
              <Download className="w-3 h-3" /> {tr.downloadTestingX}
            </button>
          </div>
          <div className="flex justify-center mt-4">
            <Button onClick={() => onGoToStep?.(5)}>
              <ChevronRight className="mr-2 h-4 w-4" /> {tr.continue5}
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
