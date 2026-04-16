import React from "react";
import { AlertTriangle, CheckCircle2, ChevronRight, Download, Lightbulb, Loader2, UploadCloud, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { FeatureSpec } from "@/lib/api";
import { downloadFeatureSpec } from "@/lib/pipelineDownloads";
import { cls, FeatureSpecBox, ProgressBar } from "./shared";
import { OllamaWarning } from "./OllamaWarning";
import type { TrainingTranslations } from "./translations";

export function DiscoveryPhasePanel(props: {
  deluxe: boolean;
  uiLanguage: "cs" | "en";
  tr: TrainingTranslations;
  targetVariable: string;
  setTargetVariable: (value: string) => void;
  targetMode: "regression" | "classification";
  setTargetMode: (value: "regression" | "classification") => void;
  discoveryFiles: File[];
  setDiscoveryFiles: React.Dispatch<React.SetStateAction<File[]>>;
  useDiscoveryLabels: boolean;
  setUseDiscoveryLabels: React.Dispatch<React.SetStateAction<boolean>>;
  discoveryLabels: File | null;
  setDiscoveryLabels: React.Dispatch<React.SetStateAction<File | null>>;
  onDiscoverStart: (files: File[], labelsFile?: File | null) => void;
  isDiscovering: boolean;
  featureSpec: FeatureSpec | null;
  setFeatureSpec: (spec: FeatureSpec) => void;
  hasDownstreamProgress: boolean;
  ollamaOk?: boolean | null;
  recheckOllama?: () => void;
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
    targetVariable,
    setTargetVariable,
    targetMode,
    setTargetMode,
    discoveryFiles,
    setDiscoveryFiles,
    useDiscoveryLabels,
    setUseDiscoveryLabels,
    discoveryLabels,
    setDiscoveryLabels,
    onDiscoverStart,
    isDiscovering,
    featureSpec,
    setFeatureSpec,
    hasDownstreamProgress,
    ollamaOk,
    recheckOllama,
    progress,
    progressLabel,
    etaText,
    onCancel,
    onGoToStep,
  } = props;

  return (
    <div className="space-y-4">
      <div>
        <label className={`block text-sm font-medium mb-1 ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
          {tr.targetVariableLabel}
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
        <p className={`mt-2 text-xs ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>
          {targetMode === "regression" ? tr.targetModeHintRegression : tr.targetModeHintClassification}
        </p>
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
              setDiscoveryFiles((prev) => [...prev, ...Array.from(e.target.files!)]);
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
            {tr.addLabelsFile}
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
            {discoveryLabels && <p className="mt-1 text-xs text-green-500 font-medium">{tr.selectedCsv}: {discoveryLabels.name}</p>}
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
          <ProgressBar deluxe={deluxe} progress={progress} label={progressLabel || tr.discoveryAnalyzing} etaText={etaText} />
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
  );
}
