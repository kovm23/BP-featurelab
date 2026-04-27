import React, { useEffect, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, ChevronRight, Cpu, X } from "lucide-react";
import { AVAILABLE_MODELS } from "@/lib/api";
import type { FeatureSpec, LlmEndpointConfig, PredictionItem, PredictionMetrics, TrainResult } from "@/lib/api";
import { cls, enrichError, QueueBusyBanner, useElapsedTimer, useProgressStall } from "./training-view/shared";
import { DiscoveryPhasePanel } from "./training-view/DiscoveryPhasePanel";
import { PredictionPhasePanel } from "./training-view/PredictionPhasePanel";
import { TestingExtractionPhasePanel } from "./training-view/TestingExtractionPhasePanel";
import { TrainingExtractionPhasePanel } from "./training-view/TrainingExtractionPhasePanel";
import { TrainingPhasePanel } from "./training-view/TrainingPhasePanel";
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
  isExtracting: boolean;
  trainingDataX: Record<string, unknown>[] | null;
  datasetYColumns: string[] | null;
  onTrain: (targetColumn: string) => void;
  isTraining: boolean;
  trainResult: TrainResult | null;
  onExtractTesting: (file: File) => void;
  isExtractingTest: boolean;
  testingDataX: Record<string, unknown>[] | null;
  onPredict: (labelsFile?: File | null) => void;
  isPredicting: boolean;
  predictions: PredictionItem[] | null;
  predictionMetrics: PredictionMetrics | null;
  modelProvider: string;
  setModelProvider: (v: string) => void;
  llmEndpoint?: LlmEndpointConfig;
  setLlmEndpoint?: (cfg: LlmEndpointConfig) => void;
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
              aria-current={isCurrent ? "step" : undefined}
              aria-label={`${isCompleted ? "✓ " : ""}Phase ${phase.num}: ${phase.short}${isCurrent ? " (current)" : ""}`}
              className={`text-xs px-2 py-1 rounded-full font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-1 ${
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
              {isCompleted && <CheckCircle2 className="inline h-3 w-3 mr-0.5 -mt-0.5" aria-hidden="true" />}
              {phase.num}. {phase.short}
            </button>
          </React.Fragment>
        );
      })}
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
  isExtracting,
  trainingDataX,
  datasetYColumns,
  onTrain,
  isTraining,
  trainResult,
  onExtractTesting,
  isExtractingTest,
  testingDataX,
  onPredict,
  isPredicting,
  predictions,
  predictionMetrics,
  modelProvider,
  setModelProvider,
  llmEndpoint,
  setLlmEndpoint,
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
  const extractStalled = useProgressStall(progress, isExtracting);
  const testExtractStalled = useProgressStall(progress, isExtractingTest);

  const errorRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (error && errorRef.current) {
      errorRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [error]);

  return (
    <div
      className={`p-6 rounded-2xl shadow-sm border ${cls(
        deluxe,
        "bg-white border-slate-100",
        "bg-slate-800/50 border-slate-700/50"
      )}`}
    >
      {error && (() => {
        const { message, hint } = enrichError(error, uiLanguage);
        return (
          <div
            ref={errorRef}
            role="alert"
            aria-live="assertive"
            className={`mb-4 flex items-start gap-2 p-3 rounded-lg border ${cls(deluxe, "bg-red-50 border-red-200 text-red-800", "bg-red-900/30 border-red-800/50 text-red-300")}`}
          >
            <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" aria-hidden="true" />
            <div className="flex-1">
              <p className="text-sm">{message}</p>
              {hint && <p className="text-xs mt-1 italic opacity-80">{hint}</p>}
              <p className="text-xs mt-1 opacity-60">
                {uiLanguage === "en" ? "Dismiss this message and try the action again." : "Zavřete tuto zprávu a zkuste akci znovu."}
              </p>
            </div>
            <button
              onClick={clearError}
              aria-label={uiLanguage === "en" ? "Dismiss error" : "Zavřít chybu"}
              className={`p-0.5 rounded ${cls(deluxe, "hover:bg-red-100", "hover:bg-red-900/50")} focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400`}
            >
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
        {phaseTitle[step] ?? (uiLanguage === "en" ? `Phase ${step}` : `Fáze ${step}`)}
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
        <DiscoveryPhasePanel
          deluxe={deluxe}
          uiLanguage={uiLanguage}
          tr={tr}
          targetVariable={targetVariable}
          setTargetVariable={setTargetVariable}
          targetMode={targetMode}
          setTargetMode={setTargetMode}
          discoveryFiles={discoveryFiles}
          setDiscoveryFiles={setDiscoveryFiles}
          useDiscoveryLabels={useDiscoveryLabels}
          setUseDiscoveryLabels={setUseDiscoveryLabels}
          discoveryLabels={discoveryLabels}
          setDiscoveryLabels={setDiscoveryLabels}
          onDiscoverStart={onDiscoverStart}
          isDiscovering={isDiscovering}
          featureSpec={featureSpec}
          setFeatureSpec={setFeatureSpec}
          hasDownstreamProgress={hasDownstreamProgress}
          ollamaOk={ollamaOk}
          recheckOllama={recheckOllama}
          progress={progress}
          progressLabel={progressLabel}
          onCancel={onCancel}
          onGoToStep={onGoToStep}
          llmEndpoint={llmEndpoint}
          setLlmEndpoint={setLlmEndpoint}
        />
      )}

      {step === 2 && (
        <TrainingExtractionPhasePanel
          deluxe={deluxe}
          uiLanguage={uiLanguage}
          tr={tr}
          featureSpec={featureSpec}
          targetVariable={targetVariable}
          setFeatureSpec={setFeatureSpec}
          trainZipFile={trainZipFile}
          setTrainZipFile={setTrainZipFile}
          useExtractionLabels={useExtractionLabels}
          setUseExtractionLabels={setUseExtractionLabels}
          extractionLabels={extractionLabels}
          setExtractionLabels={setExtractionLabels}
          onExtractTraining={onExtractTraining}
          isExtracting={isExtracting}
          trainingDataX={trainingDataX}
          ollamaOk={ollamaOk}
          recheckOllama={recheckOllama}
          progress={progress}
          progressLabel={progressLabel}
          extractStalled={extractStalled}
          onCancel={onCancel}
          onGoToStep={onGoToStep}
        />
      )}

      {step === 3 && (
        <TrainingPhasePanel
          deluxe={deluxe}
          uiLanguage={uiLanguage}
          tr={tr}
          featureSpec={featureSpec}
          targetVariable={targetVariable}
          trainingDataX={trainingDataX}
          datasetYColumns={datasetYColumns}
          targetColumn={targetColumn}
          setTargetColumn={setTargetColumn}
          onTrain={onTrain}
          isTraining={isTraining}
          trainSecs={trainSecs}
          trainResult={trainResult}
          progress={progress}
          progressLabel={progressLabel}
          onCancel={onCancel}
          onGoToStep={onGoToStep}
        />
      )}

      {step === 4 && (
        <TestingExtractionPhasePanel
          deluxe={deluxe}
          uiLanguage={uiLanguage}
          tr={tr}
          featureSpec={featureSpec}
          targetVariable={targetVariable}
          testZipFile={testZipFile}
          setTestZipFile={setTestZipFile}
          onExtractTesting={onExtractTesting}
          isExtractingTest={isExtractingTest}
          testingDataX={testingDataX}
          ollamaOk={ollamaOk}
          recheckOllama={recheckOllama}
          progress={progress}
          progressLabel={progressLabel}
          testExtractStalled={testExtractStalled}
          onCancel={onCancel}
          onGoToStep={onGoToStep}
        />
      )}

      {step === 5 && (
        <PredictionPhasePanel
          deluxe={deluxe}
          tr={tr}
          uiLanguage={uiLanguage}
          testingDataX={testingDataX}
          showPredictForm={showPredictForm}
          useTestingLabels={useTestingLabels}
          setUseTestingLabels={setUseTestingLabels}
          testingLabels={testingLabels}
          setTestingLabels={setTestingLabels}
          onPredict={onPredict}
          isPredicting={isPredicting}
          predictSecs={predictSecs}
          progress={progress}
          progressLabel={progressLabel}
          onCancel={onCancel}
          predictions={predictions}
          predictionMetrics={predictionMetrics}
          featureSpec={featureSpec}
          trainResult={trainResult}
          showFeatureCols={showFeatureCols}
          setShowFeatureCols={setShowFeatureCols}
          setShowPredictForm={setShowPredictForm}
        />
      )}
    </div>
  );
}
