import React, { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Download,
  Lightbulb,
  Loader2,
  PlayCircle,
  UploadCloud,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { AVAILABLE_MODELS } from "@/lib/api";
import type {
  FeatureSpec,
  PredictionItem,
  PredictionMetrics,
  TrainResult,
} from "@/lib/api";
import { downloadFeatureSpec, downloadTrainingDataCSV } from "@/lib/helpers";
import { PredictionResults } from "./training-view/PredictionResults";
import {
  cls,
  DatasetTable,
  enrichError,
  FeatureSpecBox,
  FileDropZone,
  formatDurationShort,
  ProgressBar,
  QueueBusyBanner,
  useElapsedTimer,
  useEstimatedRemaining,
  useProgressStall,
} from "./training-view/shared";
import { TrainingResultsCard } from "./training-view/TrainingResultsCard";
import { getTrainingTranslations } from "./training-view/translations";

export interface TrainingViewProps {
  deluxe: boolean;
  uiLanguage?: "cs" | "en";
  onDiscoverStart: (files: File[], labelsFile?: File | null) => void;
  isDiscovering: boolean;
  targetVariable: string;
  setTargetVariable: (v: string) => void;
  targetMode: "regression" | "classification";
  setTargetMode: (v: "regression" | "classification") => void;
  featureSpec: FeatureSpec | null;
  setFeatureSpec: (spec: FeatureSpec) => void;
  onExtractTraining: (file: File, labelsFile?: File | null) => void;
  onExtractTrainingLocal: (zipPath: string, labelsPath?: string) => void;
  isExtracting: boolean;
  trainingDataX: Record<string, unknown>[] | null;
  datasetYColumns: string[] | null;
  onTrain: (targetColumn: string) => void;
  isTraining: boolean;
  trainResult: TrainResult | null;
  onExtractTesting: (file: File) => void;
  onExtractTestingLocal: (zipPath: string) => void;
  isExtractingTest: boolean;
  testingDataX: Record<string, unknown>[] | null;
  onPredict: (labelsFile?: File | null) => void;
  isPredicting: boolean;
  predictions: PredictionItem[] | null;
  predictionMetrics: PredictionMetrics | null;
  modelProvider: string;
  setModelProvider: (v: string) => void;
  step: number;
  onGoToStep?: (step: number) => void;
  onCancel?: () => void;
  progress: number;
  progressLabel: string;
  error: string | null;
  clearError: () => void;
  ollamaOk?: boolean | null;
  recheckOllama?: () => void;
  queueBusy?: boolean;
  queuedCount?: number;
}

function PhaseStepper({
  deluxe,
  step,
  phaseLabels,
  onGoToStep,
  anyBusy,
  featureSpec,
  trainingDataX,
  trainResult,
  testingDataX,
  predictions,
}: {
  deluxe: boolean;
  step: number;
  phaseLabels: Array<{ num: number; short: string }>;
  onGoToStep?: (step: number) => void;
  anyBusy: boolean;
  featureSpec: FeatureSpec | null;
  trainingDataX: Record<string, unknown>[] | null;
  trainResult: TrainResult | null;
  testingDataX: Record<string, unknown>[] | null;
  predictions: PredictionItem[] | null;
}) {
  return (
    <div className="flex items-center justify-center gap-1 mb-6 flex-wrap">
      {phaseLabels.map((phase, index) => {
        const stepReachable =
          phase.num === 1 ? true :
          phase.num === 2 ? !!featureSpec :
          phase.num === 3 ? !!trainingDataX :
          phase.num === 4 ? !!trainResult :
          phase.num === 5 ? !!testingDataX :
          false;
        const isCompleted =
          phase.num === 1 ? !!featureSpec :
          phase.num === 2 ? !!trainingDataX :
          phase.num === 3 ? !!trainResult :
          phase.num === 4 ? !!testingDataX :
          phase.num === 5 ? !!predictions :
          false;
        const isCurrent = step === phase.num;
        const canClick = stepReachable && !isCurrent && onGoToStep && !anyBusy;

        return (
          <React.Fragment key={phase.num}>
            {index > 0 && (
              <ChevronRight className={`h-3.5 w-3.5 ${cls(deluxe, "text-slate-300", "text-slate-600")}`} />
            )}
            <button
              type="button"
              disabled={!canClick}
              onClick={() => canClick && onGoToStep?.(phase.num)}
              className={`text-xs px-2 py-1 rounded-full font-medium transition-colors ${
                isCurrent
                  ? "bg-blue-500 text-white"
                  : isCompleted
                    ? cls(
                        deluxe,
                        "bg-green-100 text-green-700 hover:bg-green-200",
                        "bg-green-900/40 text-green-400 hover:bg-green-900/60"
                      ) + (canClick ? " cursor-pointer" : "")
                    : cls(deluxe, "bg-slate-100 text-slate-400", "bg-slate-700 text-slate-500")
              } disabled:cursor-default`}
            >
              {isCompleted && <CheckCircle2 className="inline h-3 w-3 mr-0.5 -mt-0.5" />}
              {phase.num}. {phase.short}
            </button>
          </React.Fragment>
        );
      })}
    </div>
  );
}

function OllamaWarning({
  deluxe,
  tr,
  recheckOllama,
}: {
  deluxe: boolean;
  tr: ReturnType<typeof getTrainingTranslations>;
  recheckOllama?: () => void;
}) {
  return (
    <div className={`flex items-start gap-2 p-3 rounded-lg border ${cls(deluxe, "bg-amber-50 border-amber-300 text-amber-800", "bg-amber-900/30 border-amber-700 text-amber-300")}`}>
      <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
      <div className="flex-1">
        <p className="text-sm">
          {tr.ollamaUnavailable} <code className="font-mono bg-black/10 px-1 rounded">ollama serve</code>
        </p>
      </div>
      {recheckOllama && (
        <button onClick={recheckOllama} className="text-xs underline opacity-70 hover:opacity-100 whitespace-nowrap">
          {tr.checkAgain}
        </button>
      )}
    </div>
  );
}

export function TrainingView({
  deluxe,
  uiLanguage = "cs",
  onDiscoverStart,
  isDiscovering,
  targetVariable,
  setTargetVariable,
  targetMode,
  setTargetMode,
  featureSpec,
  setFeatureSpec,
  onExtractTraining,
  onExtractTrainingLocal,
  isExtracting,
  trainingDataX,
  datasetYColumns,
  onTrain,
  isTraining,
  trainResult,
  onExtractTesting,
  onExtractTestingLocal,
  isExtractingTest,
  testingDataX,
  onPredict,
  isPredicting,
  predictions,
  predictionMetrics,
  modelProvider,
  setModelProvider,
  step,
  onGoToStep,
  onCancel,
  progress,
  progressLabel,
  error,
  clearError,
  ollamaOk,
  recheckOllama,
  queueBusy,
  queuedCount,
}: TrainingViewProps) {
  const tr = getTrainingTranslations(uiLanguage);

  const phaseLabels = [
    { num: 1, short: tr.stepDiscovery },
    { num: 2, short: tr.stepExtraction },
    { num: 3, short: tr.stepTraining },
    { num: 4, short: tr.stepTestExtraction },
    { num: 5, short: tr.stepPrediction },
  ];

  const phaseTitle: Record<number, string> = {
    1: tr.phaseTitle1,
    2: tr.phaseTitle2,
    3: tr.phaseTitle3,
    4: tr.phaseTitle4,
    5: tr.phaseTitle5,
  };

  const phaseDesc: Record<number, string> = {
    1: tr.phaseDesc1,
    2: tr.phaseDesc2,
    3: tr.phaseDesc3,
    4: tr.phaseDesc4,
    5: tr.phaseDesc5,
  };

  const [discoveryFiles, setDiscoveryFiles] = useState<File[]>([]);
  const [trainZipFile, setTrainZipFile] = useState<File | null>(null);
  const [testZipFile, setTestZipFile] = useState<File | null>(null);
  const [useServerPathTrain, setUseServerPathTrain] = useState(false);
  const [serverPathTrain, setServerPathTrain] = useState("");
  const [serverLabelsPathTrain, setServerLabelsPathTrain] = useState("");
  const [useServerPathTest, setUseServerPathTest] = useState(false);
  const [serverPathTest, setServerPathTest] = useState("");
  const [targetColumn, setTargetColumn] = useState("");
  const [discoveryLabels, setDiscoveryLabels] = useState<File | null>(null);
  const [useDiscoveryLabels, setUseDiscoveryLabels] = useState(false);
  const [extractionLabels, setExtractionLabels] = useState<File | null>(null);
  const [useExtractionLabels, setUseExtractionLabels] = useState(false);
  const [testingLabels, setTestingLabels] = useState<File | null>(null);
  const [useTestingLabels, setUseTestingLabels] = useState(false);
  const [showPredictForm, setShowPredictForm] = useState(true);
  const [showFeatureCols, setShowFeatureCols] = useState(false);

  useEffect(() => {
    if (datasetYColumns && datasetYColumns.length > 0 && !targetColumn) {
      setTargetColumn(datasetYColumns[datasetYColumns.length - 1]);
    }
  }, [datasetYColumns]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setShowPredictForm(!predictions);
  }, [predictions]);

  const anyBusy = isDiscovering || isExtracting || isTraining || isExtractingTest || isPredicting;
  const hasDownstreamProgress = !!trainingDataX || !!trainResult || !!testingDataX || !!predictions;

  const trainSecs = useElapsedTimer(isTraining);
  const predictSecs = useElapsedTimer(isPredicting);
  const discoveryEta = useEstimatedRemaining(progress, isDiscovering);
  const extractEta = useEstimatedRemaining(progress, isExtracting);
  const trainEta = useEstimatedRemaining(progress, isTraining);
  const testExtractEta = useEstimatedRemaining(progress, isExtractingTest);
  const predictEta = useEstimatedRemaining(progress, isPredicting);
  const extractStalled = useProgressStall(progress, isExtracting);
  const testExtractStalled = useProgressStall(progress, isExtractingTest);

  const errorRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (error && errorRef.current) {
      errorRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [error]);

  const renderEtaText = (eta: number | null) =>
    eta != null ? `${tr.etaRemaining}: ~${formatDurationShort(eta)}` : null;

  return (
    <div
      className={`p-6 rounded-2xl shadow-sm border ${cls(
        deluxe,
        "bg-white border-slate-100",
        "bg-slate-800/50 border-slate-700/50"
      )}`}
    >
      {error && (() => {
        const { message, hint } = enrichError(error);
        return (
          <div ref={errorRef} className={`mb-4 flex items-start gap-2 p-3 rounded-lg border ${cls(deluxe, "bg-red-50 border-red-200 text-red-800", "bg-red-900/30 border-red-800/50 text-red-300")}`}>
            <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-sm">{message}</p>
              {hint && <p className="text-xs mt-1 italic opacity-80">{hint}</p>}
            </div>
            <button onClick={clearError} className="p-0.5 rounded hover:bg-red-100">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        );
      })()}

      {queueBusy && <QueueBusyBanner deluxe={deluxe} queuedCount={queuedCount} tr={tr} />}

      <PhaseStepper
        deluxe={deluxe}
        step={step}
        phaseLabels={phaseLabels}
        onGoToStep={onGoToStep}
        anyBusy={anyBusy}
        featureSpec={featureSpec}
        trainingDataX={trainingDataX}
        trainResult={trainResult}
        testingDataX={testingDataX}
        predictions={predictions}
      />

      <h2 className={`text-xl font-bold mb-2 ${cls(deluxe, "text-slate-900", "text-white")}`}>
        {phaseTitle[step] ?? `Fáze ${step}`}
      </h2>
      <p className={`text-sm mb-6 ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>
        {phaseDesc[step] ?? ""}
      </p>

      {(step <= 2 || step === 4) && (
        <div className="flex justify-center items-center gap-2 mb-6">
          <span className={`flex items-center gap-1.5 text-xs font-medium ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>
            <Cpu className="h-3.5 w-3.5" /> {tr.processWith}
          </span>
          <select
            value={modelProvider}
            onChange={(e) => setModelProvider(e.target.value)}
            disabled={anyBusy}
            className={`appearance-none text-xs font-bold pl-2.5 pr-7 py-1.5 rounded-md outline-none border ${cls(
              deluxe,
              "bg-white text-slate-900 border-slate-300",
              "bg-slate-800 text-white border-slate-600"
            )}`}
          >
            {AVAILABLE_MODELS.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        </div>
      )}

      {step === 1 && (
        <div className="space-y-4">
          <div>
            <label className={`block text-sm font-medium mb-1 ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
              Co chceme predikovat? (Cílová proměnná)
            </label>
            <input
              type="text"
              value={targetVariable}
              onChange={(e) => setTargetVariable(e.target.value)}
              className={`w-full p-2 text-sm rounded-lg border outline-none ${cls(
                deluxe,
                "bg-slate-50 border-slate-200 text-slate-900 focus:border-blue-400",
                "bg-slate-900 border-slate-700 text-white"
              )}`}
              placeholder="např. movie memorability score"
            />
          </div>

          <div>
            <label className={`block text-sm font-medium mb-1 ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
              {tr.targetType}
            </label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setTargetMode("regression")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
                  targetMode === "regression"
                    ? "bg-blue-500 text-white border-blue-500"
                    : cls(deluxe, "bg-white text-slate-700 border-slate-300", "bg-slate-900 text-slate-300 border-slate-700")
                }`}
              >
                {tr.regression}
              </button>
              <button
                type="button"
                onClick={() => setTargetMode("classification")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
                  targetMode === "classification"
                    ? "bg-blue-500 text-white border-blue-500"
                    : cls(deluxe, "bg-white text-slate-700 border-slate-300", "bg-slate-900 text-slate-300 border-slate-700")
                }`}
              >
                {tr.classification}
              </button>
            </div>
          </div>

          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              if (e.dataTransfer.files?.length) {
                setDiscoveryFiles((prev) => [...prev, ...Array.from(e.dataTransfer.files)]);
              }
            }}
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${cls(
              deluxe,
              "border-slate-200 hover:border-blue-400/50 bg-slate-50/50",
              "border-slate-700 hover:border-blue-500/50 bg-slate-900/50"
            )}`}
          >
            <UploadCloud className={`h-8 w-8 mx-auto mb-3 ${cls(deluxe, "text-slate-400", "text-slate-500")}`} />
            <p className={`text-sm font-medium mb-1 ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
              {uiLanguage === "en"
                ? "Upload sample media (recommended 3-5 samples for better feature proposals)"
                : "Nahrajte ukázková média (doporučeno 3-5 vzorků = lepší návrh featur)"}
            </p>
            <input
              type="file"
              id="discovery-upload"
              className="hidden"
              multiple
              onChange={(e) => {
                if (e.target.files?.length) {
                  setDiscoveryFiles((prev) => [...prev, ...Array.from(e.target.files)]);
                }
              }}
              accept=".zip,video/*,image/*,.mp4,.avi,.mov,.mkv,.png,.jpg,.jpeg"
            />
            <label htmlFor="discovery-upload" className="cursor-pointer text-blue-500 hover:text-blue-600 text-sm font-medium">
              {uiLanguage === "en" ? "Choose files from disk" : "Vybrat soubory z disku"}
            </label>
            {discoveryFiles.length > 0 && (
              <div className="mt-2 space-y-1">
                {discoveryFiles.map((file, index) => (
                  <div key={`${file.name}-${index}`} className="flex items-center justify-center gap-2 text-xs text-green-500 font-medium">
                    <span className="truncate max-w-[30ch]">{file.name}</span>
                    <button
                      onClick={() => setDiscoveryFiles((prev) => prev.filter((_, i) => i !== index))}
                      className="text-red-400 hover:text-red-600"
                      title={uiLanguage === "en" ? "Remove" : "Odebrat"}
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className={`p-3 rounded-lg border ${cls(deluxe, "bg-amber-50/50 border-amber-200", "bg-amber-900/20 border-amber-800/50")}`}>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useDiscoveryLabels}
                onChange={(e) => {
                  setUseDiscoveryLabels(e.target.checked);
                  if (!e.target.checked) setDiscoveryLabels(null);
                }}
                className="rounded"
              />
              <span className={`text-sm font-medium ${cls(deluxe, "text-amber-800", "text-amber-300")}`}>
                Přidat labels soubor (dataset_Y)
              </span>
            </label>
            {useDiscoveryLabels && (
              <div className="mt-2">
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => {
                    if (e.target.files?.[0]) setDiscoveryLabels(e.target.files[0]);
                  }}
                  className={`text-xs ${cls(deluxe, "text-slate-600", "text-slate-400")}`}
                />
                {discoveryLabels && <p className="mt-1 text-xs text-green-500 font-medium">CSV: {discoveryLabels.name}</p>}
              </div>
            )}
          </div>

          {ollamaOk === false && <OllamaWarning deluxe={deluxe} tr={tr} recheckOllama={recheckOllama} />}

          {hasDownstreamProgress && !isDiscovering && (
            <div className={`flex items-start gap-2 p-3 rounded-lg border ${cls(deluxe, "bg-amber-50 border-amber-200 text-amber-800", "bg-amber-900/20 border-amber-800/50 text-amber-300")}`}>
              <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
              <p className="text-sm">{tr.rerunDiscoveryWarning}</p>
            </div>
          )}

          <div className="flex justify-center mt-6">
            <Button
              onClick={() => discoveryFiles.length > 0 && onDiscoverStart(discoveryFiles, useDiscoveryLabels ? discoveryLabels : null)}
              disabled={discoveryFiles.length === 0 || !targetVariable || isDiscovering}
              title={[
                discoveryFiles.length === 0 && tr.uploadSamples,
                !targetVariable && tr.discoveryNeedTarget,
              ].filter(Boolean).join("; ") || undefined}
            >
              {isDiscovering ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> {tr.discoveryAnalyzing}
                </>
              ) : (
                <>
                  <Lightbulb className="mr-2 h-4 w-4" /> {tr.startDiscovery}
                </>
              )}
            </Button>
          </div>

          {isDiscovering && (
            <div className="space-y-2">
              <ProgressBar deluxe={deluxe} progress={progress} label={progressLabel || tr.discoveryAnalyzing} etaText={renderEtaText(discoveryEta)} />
              {onCancel && (
                <div className="flex justify-center">
                  <Button variant="outline" size="sm" onClick={onCancel} className="text-xs">
                    ✕ {tr.stop}
                  </Button>
                </div>
              )}
            </div>
          )}

          {featureSpec && !isDiscovering && (
            <div className={`flex items-center gap-2 p-2 rounded-lg ${cls(deluxe, "bg-green-50 text-green-700", "bg-green-900/30 text-green-400")}`}>
              <CheckCircle2 className="h-4 w-4" />
              <span className="text-sm font-medium">
                {tr.discoveryDone} — {Object.keys(featureSpec).length} {tr.featureCountProposed}
              </span>
            </div>
          )}

          {featureSpec && (
            <>
              <FeatureSpecBox
                deluxe={deluxe}
                uiLanguage={uiLanguage}
                featureSpec={featureSpec}
                targetVariable={targetVariable}
                editable
                onUpdate={setFeatureSpec}
              />
              <div className="flex justify-center mt-2">
                <button
                  onClick={() => downloadFeatureSpec(featureSpec)}
                  className="px-3 py-1.5 bg-slate-600 text-white rounded text-xs hover:bg-slate-700 flex items-center gap-1"
                >
                  <Download className="w-3 h-3" /> {tr.downloadFeatureSpec}
                </button>
              </div>
              <div className="flex justify-center mt-4">
                <Button onClick={() => onGoToStep?.(2)}>
                  <ChevronRight className="mr-2 h-4 w-4" /> {tr.continue2}
                </Button>
              </div>
            </>
          )}
        </div>
      )}

      {step === 2 && (
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

          <div className={`p-3 rounded-lg border ${cls(deluxe, "bg-slate-50 border-slate-200", "bg-slate-800/50 border-slate-700")}`}>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useServerPathTrain}
                onChange={(e) => setUseServerPathTrain(e.target.checked)}
                className="rounded"
              />
              <span className={`text-sm font-medium ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
                {tr.uploadAlreadyOnServer}
              </span>
            </label>
          </div>

          {!useServerPathTrain ? (
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
          ) : (
            <div className="space-y-2">
              <label className={`text-sm font-medium ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>{tr.serverZipPath}</label>
              <input
                type="text"
                value={serverPathTrain}
                onChange={(e) => setServerPathTrain(e.target.value)}
                placeholder="/home/kovm23/train.zip"
                className={`w-full px-3 py-2 rounded border text-sm font-mono ${cls(deluxe, "bg-white border-slate-300 text-slate-800", "bg-slate-900 border-slate-600 text-slate-200")}`}
              />
              <label className={`text-sm font-medium ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>{tr.serverLabelsPath}</label>
              <input
                type="text"
                value={serverLabelsPathTrain}
                onChange={(e) => setServerLabelsPathTrain(e.target.value)}
                placeholder={tr.serverLabelsPathPlaceholder}
                className={`w-full px-3 py-2 rounded border text-sm font-mono ${cls(deluxe, "bg-white border-slate-300 text-slate-800", "bg-slate-900 border-slate-600 text-slate-200")}`}
              />
            </div>
          )}

          {!useServerPathTrain && (
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
                  {extractionLabels && <p className="mt-1 text-xs text-green-500 font-medium">CSV: {extractionLabels.name}</p>}
                </div>
              )}
            </div>
          )}

          {ollamaOk === false && <OllamaWarning deluxe={deluxe} tr={tr} recheckOllama={recheckOllama} />}

          <div className="flex justify-center mt-6">
            <Button
              onClick={() => {
                if (useServerPathTrain) {
                  onExtractTrainingLocal(serverPathTrain, serverLabelsPathTrain || undefined);
                } else if (trainZipFile) {
                  onExtractTraining(trainZipFile, useExtractionLabels ? extractionLabels : null);
                }
              }}
              disabled={(useServerPathTrain ? !serverPathTrain : !trainZipFile) || isExtracting}
              title={!useServerPathTrain && !trainZipFile ? tr.uploadTrainingFirst : undefined}
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
              <ProgressBar deluxe={deluxe} progress={progress} label={progressLabel} etaText={renderEtaText(extractEta)} />
              {extractStalled && (
                <p className={`text-xs ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>
                  ℹ {tr.processingMayTakeLong}
                </p>
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
              <DatasetTable deluxe={deluxe} data={trainingDataX} title="Training Dataset X" />
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
      )}

      {step === 3 && (
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
              <ProgressBar deluxe={deluxe} progress={progress} label={progressLabel || tr.trainingInProgressLabel} etaText={renderEtaText(trainEta)} />
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
      )}

      {step === 4 && (
        <div className="space-y-4">
          {featureSpec && (
            <FeatureSpecBox deluxe={deluxe} uiLanguage={uiLanguage} featureSpec={featureSpec} targetVariable={targetVariable} />
          )}

          <div className={`p-3 rounded-lg border ${cls(deluxe, "bg-slate-50 border-slate-200", "bg-slate-800/50 border-slate-700")}`}>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useServerPathTest}
                onChange={(e) => setUseServerPathTest(e.target.checked)}
                className="rounded"
              />
              <span className={`text-sm font-medium ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
                {tr.uploadAlreadyOnServer}
              </span>
            </label>
          </div>

          {!useServerPathTest ? (
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
          ) : (
            <div className="space-y-2">
              <label className={`text-sm font-medium ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>{tr.serverZipPath}</label>
              <input
                type="text"
                value={serverPathTest}
                onChange={(e) => setServerPathTest(e.target.value)}
                placeholder="/home/kovm23/test.zip"
                className={`w-full px-3 py-2 rounded border text-sm font-mono ${cls(deluxe, "bg-white border-slate-300 text-slate-800", "bg-slate-900 border-slate-600 text-slate-200")}`}
              />
            </div>
          )}

          {ollamaOk === false && <OllamaWarning deluxe={deluxe} tr={tr} recheckOllama={recheckOllama} />}

          <div className="flex justify-center mt-6">
            <Button
              onClick={() => {
                if (useServerPathTest) {
                  onExtractTestingLocal(serverPathTest);
                } else if (testZipFile) {
                  onExtractTesting(testZipFile);
                }
              }}
              disabled={(useServerPathTest ? !serverPathTest : !testZipFile) || isExtractingTest}
              title={!useServerPathTest && !testZipFile ? tr.uploadTestingFirst : undefined}
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
              <ProgressBar deluxe={deluxe} progress={progress} label={progressLabel} etaText={renderEtaText(testExtractEta)} />
              {testExtractStalled && (
                <p className={`text-xs ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>
                  ℹ {tr.processingMayTakeLong}
                </p>
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
              <DatasetTable deluxe={deluxe} data={testingDataX} title="Testing Dataset X" />
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
      )}

      {step === 5 && (
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
              <ProgressBar deluxe={deluxe} progress={progress} label={progressLabel || tr.predictionInProgressLabel} etaText={renderEtaText(predictEta)} />
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
      )}
    </div>
  );
}
