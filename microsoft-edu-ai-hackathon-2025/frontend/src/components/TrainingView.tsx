import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import {
  UploadCloud,
  Lightbulb,
  PlayCircle,
  Loader2,
  Cpu,
  Download,
  CheckCircle2,
  ChevronRight,
  AlertTriangle,
  X,
  Plus,
  Trash2,
  Pencil,
} from "lucide-react";
import type { TrainResult, PredictionItem, PredictionMetrics } from "@/lib/api";
import { AVAILABLE_MODELS } from "@/lib/api";
import {
  downloadFeatureSpec,
  downloadTrainingDataCSV,
  downloadRulesModel,
  downloadExperimentZip,
} from "@/lib/helpers";

/* ------------------------------------------------------------------ */
/*  Typy                                                               */
/* ------------------------------------------------------------------ */

export interface TrainingViewProps {
  deluxe: boolean;
  /* Phase 1 */
  onDiscoverStart: (files: File[], labelsFile?: File | null) => void;
  isDiscovering: boolean;
  targetVariable: string;
  setTargetVariable: (v: string) => void;
  featureSpec: Record<string, string> | null;
  setFeatureSpec: (spec: Record<string, string>) => void;
  /* Phase 2 */
  onExtractTraining: (file: File, labelsFile?: File | null) => void;
  onExtractTrainingLocal: (zipPath: string, labelsPath?: string) => void;
  isExtracting: boolean;
  trainingDataX: Record<string, unknown>[] | null;
  datasetYColumns: string[] | null;
  /* Phase 3 */
  onTrain: (targetColumn: string) => void;
  isTraining: boolean;
  trainResult: TrainResult | null;
  /* Phase 4 */
  onExtractTesting: (file: File) => void;
  onExtractTestingLocal: (zipPath: string) => void;
  isExtractingTest: boolean;
  testingDataX: Record<string, unknown>[] | null;
  /* Phase 5 */
  onPredict: (labelsFile?: File | null) => void;
  isPredicting: boolean;
  predictions: PredictionItem[] | null;
  predictionMetrics: PredictionMetrics | null;
  /* Common */
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
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

const PHASE_LABELS = [
  { num: 1, short: "Discovery" },
  { num: 2, short: "Extrakce" },
  { num: 3, short: "Trénink" },
  { num: 4, short: "Test ext." },
  { num: 5, short: "Predikce" },
];

function cls(deluxe: boolean, light: string, dark: string) {
  return deluxe ? dark : light;
}

/* ------------------------------------------------------------------ */
/*  Reusable: drop-zone + file picker                                  */
/* ------------------------------------------------------------------ */

function FileDropZone({
  deluxe,
  file,
  onFile,
  accept,
  inputId,
  label,
  pickLabel,
}: {
  deluxe: boolean;
  file: File | null;
  onFile: (f: File) => void;
  accept: string;
  inputId: string;
  label: string;
  pickLabel: string;
}) {
  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        if (e.dataTransfer.files?.[0]) onFile(e.dataTransfer.files[0]);
      }}
      className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${cls(
        deluxe,
        "border-slate-200 hover:border-blue-400/50 bg-slate-50/50",
        "border-slate-700 hover:border-blue-500/50 bg-slate-900/50"
      )}`}
    >
      <UploadCloud
        className={`h-8 w-8 mx-auto mb-3 ${cls(
          deluxe,
          "text-slate-400",
          "text-slate-500"
        )}`}
      />
      <p
        className={`text-sm font-medium mb-1 ${cls(
          deluxe,
          "text-slate-700",
          "text-slate-300"
        )}`}
      >
        {label}
      </p>
      <input
        type="file"
        id={inputId}
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.[0]) onFile(e.target.files[0]);
        }}
        accept={accept}
      />
      <label
        htmlFor={inputId}
        className="cursor-pointer text-blue-500 hover:text-blue-600 text-sm font-medium"
      >
        {pickLabel}
      </label>
      {file && (
        <p className="mt-2 text-xs text-green-500 font-medium truncate">
          Vybráno: {file.name}
        </p>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Reusable: progress bar                                             */
/* ------------------------------------------------------------------ */

function ProgressBar({
  deluxe,
  progress,
  label,
}: {
  deluxe: boolean;
  progress: number;
  label: string;
}) {
  return (
    <div className="mt-6 space-y-2">
      <div className="flex justify-between text-xs font-medium">
        <span className={cls(deluxe, "text-slate-500", "text-slate-400")}>
          {label}
        </span>
        <span className={cls(deluxe, "text-slate-700", "text-slate-300")}>
          {progress}%
        </span>
      </div>
      <div
        className={`h-2 rounded-full overflow-hidden ${cls(
          deluxe,
          "bg-slate-100",
          "bg-slate-800"
        )}`}
      >
        <div
          className="h-full bg-blue-500 transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Reusable: feature spec editor                                      */
/* ------------------------------------------------------------------ */

function FeatureSpecBox({
  deluxe,
  featureSpec,
  targetVariable,
  editable = false,
  onUpdate,
}: {
  deluxe: boolean;
  featureSpec: Record<string, string>;
  targetVariable: string;
  editable?: boolean;
  onUpdate?: (spec: Record<string, string>) => void;
}) {
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [addingNew, setAddingNew] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");

  const startEdit = (key: string, desc: string) => {
    setEditingKey(key);
    setEditName(key);
    setEditDesc(desc);
  };

  const saveEdit = () => {
    if (!editingKey || !editName.trim() || !onUpdate) return;
    const updated = { ...featureSpec };
    if (editName !== editingKey) delete updated[editingKey];
    updated[editName.trim()] = editDesc.trim();
    onUpdate(updated);
    setEditingKey(null);
  };

  const removeFeature = (key: string) => {
    if (!onUpdate) return;
    const updated = { ...featureSpec };
    delete updated[key];
    onUpdate(updated);
  };

  const addFeature = () => {
    if (!newName.trim() || !onUpdate) return;
    onUpdate({ ...featureSpec, [newName.trim()]: newDesc.trim() });
    setNewName("");
    setNewDesc("");
    setAddingNew(false);
  };

  return (
    <div
      className={`p-4 rounded-lg border ${cls(
        deluxe,
        "bg-blue-50 border-blue-100",
        "bg-slate-900 border-slate-700"
      )}`}
    >
      <div className="flex items-center justify-between mb-2">
        <h3
          className={`text-sm font-bold ${cls(
            deluxe,
            "text-blue-900",
            "text-white"
          )}`}
        >
          Feature Definition Spec (Cíl: {targetVariable}):
        </h3>
        {editable && (
          <span className={`text-[10px] px-1.5 py-0.5 rounded ${cls(deluxe, "bg-amber-100 text-amber-700", "bg-amber-900/40 text-amber-400")}`}>
            Editovatelné
          </span>
        )}
      </div>
      <ul
        className={`text-xs space-y-1.5 ${cls(
          deluxe,
          "text-slate-700",
          "text-slate-300"
        )}`}
      >
        {Object.entries(featureSpec).map(([key, desc]) =>
          editable && editingKey === key ? (
            <li key={key} className="flex flex-col gap-1 p-2 rounded bg-black/5">
              <input
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                className={`text-xs font-mono px-1.5 py-0.5 rounded border outline-none ${cls(deluxe, "bg-white border-slate-300", "bg-slate-800 border-slate-600 text-white")}`}
                placeholder="Název feature"
              />
              <input
                value={editDesc}
                onChange={(e) => setEditDesc(e.target.value)}
                className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(deluxe, "bg-white border-slate-300", "bg-slate-800 border-slate-600 text-white")}`}
                placeholder="Popis"
              />
              <div className="flex gap-1">
                <button onClick={saveEdit} className="text-[10px] px-1.5 py-0.5 rounded bg-green-500 text-white hover:bg-green-600">Uložit</button>
                <button onClick={() => setEditingKey(null)} className="text-[10px] px-1.5 py-0.5 rounded bg-slate-400 text-white hover:bg-slate-500">Zrušit</button>
              </div>
            </li>
          ) : (
            <li key={key} className="flex items-center gap-2 group">
              <span className="font-mono bg-black/10 px-1 rounded">{key}</span>:{" "}
              <span className="flex-1">{String(desc)}</span>
              {editable && (
                <span className="flex gap-0.5 opacity-60 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
                  <button onClick={() => startEdit(key, desc)} title="Upravit" className="p-0.5 rounded hover:bg-black/10">
                    <Pencil className="h-3 w-3" />
                  </button>
                  <button onClick={() => removeFeature(key)} title="Odebrat" className="p-0.5 rounded hover:bg-red-100 text-red-500">
                    <Trash2 className="h-3 w-3" />
                  </button>
                </span>
              )}
            </li>
          )
        )}
      </ul>

      {editable && (
        <div className="mt-2">
          {addingNew ? (
            <div className="flex flex-col gap-1 p-2 rounded bg-black/5">
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className={`text-xs font-mono px-1.5 py-0.5 rounded border outline-none ${cls(deluxe, "bg-white border-slate-300", "bg-slate-800 border-slate-600 text-white")}`}
                placeholder="Název nové feature (např. scene_brightness)"
              />
              <input
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(deluxe, "bg-white border-slate-300", "bg-slate-800 border-slate-600 text-white")}`}
                placeholder="Popis (např. průměrný jas scény 0-255)"
              />
              <div className="flex gap-1">
                <button onClick={addFeature} disabled={!newName.trim()} className="text-[10px] px-1.5 py-0.5 rounded bg-green-500 text-white hover:bg-green-600 disabled:opacity-50">Přidat</button>
                <button onClick={() => setAddingNew(false)} className="text-[10px] px-1.5 py-0.5 rounded bg-slate-400 text-white hover:bg-slate-500">Zrušit</button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setAddingNew(true)}
              className={`flex items-center gap-1 text-[10px] mt-1 px-2 py-0.5 rounded hover:bg-black/10 ${cls(deluxe, "text-blue-600", "text-blue-400")}`}
            >
              <Plus className="h-3 w-3" /> Přidat feature
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Reusable: dataset_X table preview                                  */
/* ------------------------------------------------------------------ */

function DatasetTable({
  deluxe,
  data,
  title,
}: {
  deluxe: boolean;
  data: Record<string, unknown>[];
  title: string;
}) {
  if (!data.length) return null;
  const cols = Object.keys(data[0]);

  return (
    <div className={`rounded-lg border overflow-hidden ${cls(deluxe, "border-slate-200", "border-slate-700")}`}>
      <p className={`text-xs font-bold px-3 py-2 ${cls(deluxe, "bg-slate-100 text-slate-700", "bg-slate-800 text-slate-300")}`}>
        {title} ({data.length} řádků)
      </p>
      <div className="overflow-x-auto max-h-60">
        <table className="w-full text-xs">
          <thead>
            <tr className={cls(deluxe, "bg-slate-50", "bg-slate-900")}>
              {cols.map((c) => (
                <th key={c} className={`px-2 py-1 text-left font-mono whitespace-nowrap ${cls(deluxe, "text-slate-600", "text-slate-400")}`}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i} className={i % 2 === 0 ? cls(deluxe, "bg-white", "bg-slate-800/50") : cls(deluxe, "bg-slate-50/50", "bg-slate-900/50")}>
                {cols.map((c) => (
                  <td key={c} className={`px-2 py-1 whitespace-nowrap ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
                    {String(row[c] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Hook: elapsed time counter (for phases without progress bar)      */
/* ------------------------------------------------------------------ */

function useElapsedTimer(active: boolean) {
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    if (!active) { setSecs(0); return; }
    setSecs(0);
    const id = setInterval(() => setSecs((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [active]);
  return secs;
}

/* ------------------------------------------------------------------ */
/*  Hook: stall detection — true when progress is stuck > threshold   */
/* ------------------------------------------------------------------ */

function useProgressStall(progress: number, active: boolean, thresholdMs = 90_000) {
  const [stalled, setStalled] = useState(false);
  const lastRef = useRef({ val: -1, time: 0 });

  useEffect(() => {
    if (!active) { setStalled(false); return; }
    if (progress !== lastRef.current.val) {
      lastRef.current = { val: progress, time: Date.now() };
      setStalled(false);
      return;
    }
    const id = setInterval(() => {
      if (Date.now() - lastRef.current.time > thresholdMs) setStalled(true);
    }, 5000);
    return () => clearInterval(id);
  }, [progress, active, thresholdMs]);

  return stalled;
}

/* ------------------------------------------------------------------ */
/*  Helper: enrich backend error with actionable hint                  */
/* ------------------------------------------------------------------ */

function enrichError(raw: string): { message: string; hint?: string } {
  if (raw.includes("No data remained after joining")) {
    return {
      message: raw,
      hint: 'Zkontroluj, že názvy souborů v CSV (první sloupec bez přípony) odpovídají názvům médií v ZIPu. Např. soubor "video.mp4" → řádek CSV musí mít hodnotu "video".',
    };
  }
  if (raw.includes("Column '") && raw.includes("not found")) {
    return {
      message: raw,
      hint: "Dostupné sloupce jsou vypsány v chybě výše — zkopíruj přesný název.",
    };
  }
  if (raw.includes("ZIP contains no media files")) {
    return {
      message: raw,
      hint: "ZIP musí obsahovat videa (.mp4, .avi, .mov, .mkv) nebo obrázky (.jpg, .png, .webp, .gif).",
    };
  }
  if (raw.includes("Phase 2") && raw.includes("must be completed")) {
    return { message: raw, hint: "Vrať se na Fázi 2 a dokonči extrakci trénovacích dat." };
  }
  if (raw.includes("Model is not trained")) {
    return { message: raw, hint: "Nejprve dokonči Fázi 3 (Trénink)." };
  }
  if (raw.includes("Missing dataset_Y")) {
    return {
      message: raw,
      hint: "CSV s labels musí být součástí ZIPu nebo ho nahraj samostatně přes checkbox níže.",
    };
  }
  return { message: raw };
}

/* ================================================================== */
/*  HLAVNÍ KOMPONENTA                                                  */
/* ================================================================== */

export function TrainingView({
  deluxe,
  onDiscoverStart,
  isDiscovering,
  targetVariable,
  setTargetVariable,
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
}: TrainingViewProps) {
  const [discoveryFiles, setDiscoveryFiles] = useState<File[]>([]);
  const [trainZipFile, setTrainZipFile] = useState<File | null>(null);
  const [testZipFile, setTestZipFile] = useState<File | null>(null);
  const [useServerPathTrain, setUseServerPathTrain] = useState(false);
  const [serverPathTrain, setServerPathTrain] = useState("/home/kovm23/train_split.zip");
  const [serverLabelsPathTrain, setServerLabelsPathTrain] = useState("");
  const [useServerPathTest, setUseServerPathTest] = useState(false);
  const [serverPathTest, setServerPathTest] = useState("/home/kovm23/test_split.zip");
  const [targetColumn, setTargetColumn] = useState("");
  // Optional labels files
  const [discoveryLabels, setDiscoveryLabels] = useState<File | null>(null);
  const [useDiscoveryLabels, setUseDiscoveryLabels] = useState(false);
  const [extractionLabels, setExtractionLabels] = useState<File | null>(null);
  const [useExtractionLabels, setUseExtractionLabels] = useState(false);
  const [testingLabels, setTestingLabels] = useState<File | null>(null);
  const [useTestingLabels, setUseTestingLabels] = useState(false);

  const phaseTitle: Record<number, string> = {
    1: "Fáze 1: Feature Discovery",
    2: "Fáze 2: Feature Inference (Extrakce)",
    3: "Fáze 3: Trénink ML modelu (RuleKit)",
    4: "Fáze 4: Test Data Feature Inference",
    5: "Fáze 5: Predikce",
  };

  const phaseDesc: Record<number, string> = {
    1: "Nahrajte ZIP s ukázkovými médii (nebo jedno video/obrázek) a popište cílovou proměnnou. " +
       "AI analyzuje vzorky a navrhne Feature Definition Spec — slovník feature_name → popis s rozsahem/jednotkou.",
    2: "Zkontrolujte a upravte Feature Spec z Fáze 1. Poté nahrajte trénovací ZIP (média + CSV s labels). " +
       "AI extrahuje hodnoty features pro každý soubor → vznikne training dataset_X.",
    3: "Vyberte sloupec s cílovou proměnnou z CSV. RuleKit natrénuje pravidlový model z dataset_X + dataset_Y.",
    4: "Nahrajte testovací ZIP dataset (jiná média než trénovací!). " +
       "AI extrahuje features stejným Feature Specem → vznikne testing dataset_X.",
    5: "Model predikuje label pro každý testovací objekt. " +
       "U každé predikce vidíte: predikovaný score a použité IF-THEN pravidlo.",
  };

  const anyBusy = isDiscovering || isExtracting || isTraining || isExtractingTest || isPredicting;

  // Elapsed timers for phases without a progress bar
  const trainSecs = useElapsedTimer(isTraining);
  const predictSecs = useElapsedTimer(isPredicting);

  // Stall detection for long extractions
  const extractStalled = useProgressStall(progress, isExtracting);
  const testExtractStalled = useProgressStall(progress, isExtractingTest);

  return (
    <div
      className={`p-6 rounded-2xl shadow-sm border ${cls(
        deluxe,
        "bg-white border-slate-100",
        "bg-slate-800/50 border-slate-700/50"
      )}`}
    >
      {/* ---- ERROR BANNER ---- */}
      {error && (() => {
        const { message, hint } = enrichError(error);
        return (
        <div className={`mb-4 flex items-start gap-2 p-3 rounded-lg border ${cls(deluxe, "bg-red-50 border-red-200 text-red-800", "bg-red-900/30 border-red-800/50 text-red-300")}`}>
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

      {/* ---- STEPPER ---- */}
      <div className="flex items-center justify-center gap-1 mb-6 flex-wrap">
        {PHASE_LABELS.map((p, i) => {
          // Determine which steps have data and are reachable
          const stepHasData =
            p.num === 1 ? true :
            p.num === 2 ? !!featureSpec :
            p.num === 3 ? !!trainingDataX :
            p.num === 4 ? !!trainResult :
            p.num === 5 ? !!testingDataX :
            false;
          const isCompleted = step > p.num || (stepHasData && step !== p.num);
          const isCurrent = step === p.num;
          const canClick = stepHasData && !isCurrent && onGoToStep && !anyBusy;

          return (
            <React.Fragment key={p.num}>
              {i > 0 && (
                <ChevronRight
                  className={`h-3.5 w-3.5 ${cls(
                    deluxe,
                    "text-slate-300",
                    "text-slate-600"
                  )}`}
                />
              )}
              <button
                type="button"
                disabled={!canClick}
                onClick={() => canClick && onGoToStep!(p.num)}
                className={`text-xs px-2 py-1 rounded-full font-medium transition-colors ${
                  isCurrent
                    ? "bg-blue-500 text-white"
                    : isCompleted
                      ? cls(
                          deluxe,
                          "bg-green-100 text-green-700 hover:bg-green-200",
                          "bg-green-900/40 text-green-400 hover:bg-green-900/60"
                        ) + (canClick ? " cursor-pointer" : "")
                      : cls(
                          deluxe,
                          "bg-slate-100 text-slate-400",
                          "bg-slate-700 text-slate-500"
                        )
                } disabled:cursor-default`}
              >
                {isCompleted && <CheckCircle2 className="inline h-3 w-3 mr-0.5 -mt-0.5" />}
                {p.num}. {p.short}
              </button>
            </React.Fragment>
          );
        })}
      </div>

      {/* ---- TITLE + DESC ---- */}
      <h2
        className={`text-xl font-bold mb-2 ${cls(
          deluxe,
          "text-slate-900",
          "text-white"
        )}`}
      >
        {phaseTitle[step] ?? `Fáze ${step}`}
      </h2>
      <p
        className={`text-sm mb-6 ${cls(
          deluxe,
          "text-slate-500",
          "text-slate-400"
        )}`}
      >
        {phaseDesc[step] ?? ""}
      </p>

      {/* ---- MODEL PICKER ---- */}
      {(step <= 2 || step === 4) && (
        <div className="flex justify-center items-center gap-2 mb-6">
          <span
            className={`flex items-center gap-1.5 text-xs font-medium ${cls(
              deluxe,
              "text-slate-500",
              "text-slate-400"
            )}`}
          >
            <Cpu className="h-3.5 w-3.5" /> Zpracovat pomocí:
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
            {AVAILABLE_MODELS.map((m) => (
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
          </select>
        </div>
      )}

      {/* ================================================================ */}
      {/*  FÁZE 1: FEATURE DISCOVERY                                       */}
      {/* ================================================================ */}
      {step === 1 && (
        <div className="space-y-4">
          <div>
            <label
              className={`block text-sm font-medium mb-1 ${cls(
                deluxe,
                "text-slate-700",
                "text-slate-300"
              )}`}
            >
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
              Nahrajte ukázková média (doporučeno 3–5 vzorků = lepší návrh featur)
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
            <label
              htmlFor="discovery-upload"
              className="cursor-pointer text-blue-500 hover:text-blue-600 text-sm font-medium"
            >
              Vybrat soubory z disku
            </label>
            {discoveryFiles.length > 0 && (
              <div className="mt-2 space-y-1">
                {discoveryFiles.map((f, i) => (
                  <div key={i} className="flex items-center justify-center gap-2 text-xs text-green-500 font-medium">
                    <span className="truncate max-w-[30ch]">{f.name}</span>
                    <button
                      onClick={() => setDiscoveryFiles((prev) => prev.filter((_, j) => j !== i))}
                      className="text-red-400 hover:text-red-600"
                      title="Odebrat"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Optional labels for discovery */}
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
                Use labels for this phase (dataset_Y)
              </span>
            </label>
            {useDiscoveryLabels && (
              <div className="mt-2">
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => { if (e.target.files?.[0]) setDiscoveryLabels(e.target.files[0]); }}
                  className={`text-xs ${cls(deluxe, "text-slate-600", "text-slate-400")}`}
                />
                {discoveryLabels && (
                  <p className="mt-1 text-xs text-green-500 font-medium">CSV: {discoveryLabels.name}</p>
                )}
              </div>
            )}
          </div>

          {ollamaOk === false && (
            <div className={`flex items-start gap-2 p-3 rounded-lg border ${cls(deluxe, "bg-amber-50 border-amber-300 text-amber-800", "bg-amber-900/30 border-amber-700 text-amber-300")}`}>
              <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <p className="text-sm">Ollama není dostupný (localhost:11434). Spusťte: <code className="font-mono bg-black/10 px-1 rounded">ollama serve</code></p>
              </div>
              {recheckOllama && (
                <button onClick={recheckOllama} className="text-xs underline opacity-70 hover:opacity-100 whitespace-nowrap">Zkontrolovat znovu</button>
              )}
            </div>
          )}

          <div className="flex justify-center mt-6">
            <Button
              onClick={() => discoveryFiles.length > 0 && onDiscoverStart(discoveryFiles, useDiscoveryLabels ? discoveryLabels : null)}
              disabled={discoveryFiles.length === 0 || !targetVariable || isDiscovering}
              title={discoveryFiles.length === 0 ? "Nejdříve nahrajte vzorková média" : !targetVariable ? "Zadejte cílovou proměnnou" : undefined}
            >
              {isDiscovering ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> AI analyzuje vzorky...
                </>
              ) : (
                <>
                  <Lightbulb className="mr-2 h-4 w-4" /> Spustit Feature Discovery
                </>
              )}
            </Button>
          </div>

          {/* Feature spec výsledek + pokračovat */}
          {featureSpec && !isDiscovering && (
            <div className={`flex items-center gap-2 p-2 rounded-lg ${cls(deluxe, "bg-green-50 text-green-700", "bg-green-900/30 text-green-400")}`}>
              <CheckCircle2 className="h-4 w-4" />
              <span className="text-sm font-medium">Feature Discovery dokončena — {Object.keys(featureSpec).length} featur navrženo</span>
            </div>
          )}
          {featureSpec && (
            <>
              <FeatureSpecBox
                deluxe={deluxe}
                featureSpec={featureSpec}
                targetVariable={targetVariable}
                editable
                onUpdate={setFeatureSpec}
              />
              <div className="flex justify-center mt-4">
                <Button onClick={() => onGoToStep?.(2)}>
                  <ChevronRight className="mr-2 h-4 w-4" /> Pokračovat na Fázi 2
                </Button>
              </div>
            </>
          )}
        </div>
      )}

      {/* ================================================================ */}
      {/*  FÁZE 2: FEATURE EXTRACTION (TRAINING)                           */}
      {/* ================================================================ */}
      {step === 2 && (
        <div className="space-y-4">
          {featureSpec && (
            <FeatureSpecBox
              deluxe={deluxe}
              featureSpec={featureSpec}
              targetVariable={targetVariable}
              editable
              onUpdate={setFeatureSpec}
            />
          )}

          {/* Toggle: upload vs server path */}
          <div className={`p-3 rounded-lg border ${cls(deluxe, "bg-slate-50 border-slate-200", "bg-slate-800/50 border-slate-700")}`}>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useServerPathTrain}
                onChange={(e) => setUseServerPathTrain(e.target.checked)}
                className="rounded"
              />
              <span className={`text-sm font-medium ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
                Soubor je již na serveru (velký dataset)
              </span>
            </label>
          </div>

          {!useServerPathTrain ? (
            <FileDropZone
              deluxe={deluxe}
              file={trainZipFile}
              onFile={setTrainZipFile}
              accept=".zip"
              inputId="train-zip-upload"
              label="Nahrajte .ZIP trénovací dataset (média + CSV s labels)"
              pickLabel="Vybrat ZIP soubor"
            />
          ) : (
            <div className="space-y-2">
              <label className={`text-sm font-medium ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
                Cesta k ZIP souboru na serveru
              </label>
              <input
                type="text"
                value={serverPathTrain}
                onChange={(e) => setServerPathTrain(e.target.value)}
                placeholder="/home/kovm23/train.zip"
                className={`w-full px-3 py-2 rounded border text-sm font-mono ${cls(deluxe, "bg-white border-slate-300 text-slate-800", "bg-slate-900 border-slate-600 text-slate-200")}`}
              />
              <label className={`text-sm font-medium ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
                Cesta k CSV labels (volitelné)
              </label>
              <input
                type="text"
                value={serverLabelsPathTrain}
                onChange={(e) => setServerLabelsPathTrain(e.target.value)}
                placeholder="(volitelné, CSV je součástí ZIPu)"
                className={`w-full px-3 py-2 rounded border text-sm font-mono ${cls(deluxe, "bg-white border-slate-300 text-slate-800", "bg-slate-900 border-slate-600 text-slate-200")}`}
              />
            </div>
          )}

          {/* Optional separate labels for extraction (only for upload mode) */}
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
                  Use labels for this phase (dataset_Y)
                </span>
              </label>
              <p className={`text-xs mt-1 ${cls(deluxe, "text-amber-600", "text-amber-400/70")}`}>
                Pokud je CSV s labels v ZIPu, nahrajeme ho automaticky. Zde ho lze nahrát zvlášť.
              </p>
              {useExtractionLabels && (
                <div className="mt-2">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => { if (e.target.files?.[0]) setExtractionLabels(e.target.files[0]); }}
                    className={`text-xs ${cls(deluxe, "text-slate-600", "text-slate-400")}`}
                  />
                  {extractionLabels && (
                    <p className="mt-1 text-xs text-green-500 font-medium">CSV: {extractionLabels.name}</p>
                  )}
                </div>
              )}
            </div>
          )}

          {ollamaOk === false && (
            <div className={`flex items-start gap-2 p-3 rounded-lg border ${cls(deluxe, "bg-amber-50 border-amber-300 text-amber-800", "bg-amber-900/30 border-amber-700 text-amber-300")}`}>
              <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <p className="text-sm">Ollama není dostupný (localhost:11434). Spusťte: <code className="font-mono bg-black/10 px-1 rounded">ollama serve</code></p>
              </div>
              {recheckOllama && (
                <button onClick={recheckOllama} className="text-xs underline opacity-70 hover:opacity-100 whitespace-nowrap">Zkontrolovat znovu</button>
              )}
            </div>
          )}

          <div className="flex justify-center mt-6">
            <Button
              onClick={() => {
                if (useServerPathTrain) {
                  onExtractTrainingLocal(serverPathTrain, serverLabelsPathTrain || undefined);
                } else {
                  trainZipFile && onExtractTraining(trainZipFile, useExtractionLabels ? extractionLabels : null);
                }
              }}
              disabled={(useServerPathTrain ? !serverPathTrain : !trainZipFile) || isExtracting}
              title={!useServerPathTrain && !trainZipFile ? "Nejdříve nahrajte trénovací ZIP" : undefined}
            >
              {isExtracting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Extrahuji features...
                </>
              ) : (
                <>
                  <PlayCircle className="mr-2 h-4 w-4" /> Spustit extrakci features
                </>
              )}
            </Button>
          </div>

          {isExtracting && (
            <div className="space-y-2">
              <ProgressBar deluxe={deluxe} progress={progress} label={progressLabel} />
              {extractStalled && (
                <p className={`text-xs ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>
                  ℹ Zpracování probíhá — větší soubory nebo pomalý model mohou trvat i 10+ minut.
                </p>
              )}
              <div className="flex gap-2">
                {onCancel && (
                  <Button variant="outline" size="sm" onClick={onCancel} className="text-xs">
                    ✕ Zastavit
                  </Button>
                )}
              </div>
            </div>
          )}

          {/* Výsledek extrakce + pokračovat */}
          {trainingDataX && !isExtracting && (
            <div className={`flex items-center gap-2 p-2 rounded-lg ${cls(deluxe, "bg-green-50 text-green-700", "bg-green-900/30 text-green-400")}`}>
              <CheckCircle2 className="h-4 w-4" />
              <span className="text-sm font-medium">Extrakce dokončena — {trainingDataX.length} řádků</span>
            </div>
          )}
          {trainingDataX && (
            <>
              <DatasetTable deluxe={deluxe} data={trainingDataX} title="Training Dataset X" />
              <div className="flex justify-center mt-4">
                <Button onClick={() => onGoToStep?.(3)}>
                  <ChevronRight className="mr-2 h-4 w-4" /> Pokračovat na Fázi 3
                </Button>
              </div>
            </>
          )}
        </div>
      )}

      {/* ================================================================ */}
      {/*  FÁZE 3: ML TRAINING (RULEKIT)                                   */}
      {/* ================================================================ */}
      {step === 3 && (
        <div className="space-y-4">
          {featureSpec && (
            <FeatureSpecBox
              deluxe={deluxe}
              featureSpec={featureSpec}
              targetVariable={targetVariable}
            />
          )}

          {/* Training dataset_X preview */}
          {trainingDataX && (
            <DatasetTable deluxe={deluxe} data={trainingDataX} title="Training Dataset X" />
          )}

          {/* Target column selector */}
          <div>
            <label className={`block text-sm font-medium mb-1 ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
              Název sloupce s cílovou proměnnou (z CSV v ZIPu):
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
                <option value="">-- vyberte sloupec --</option>
                {datasetYColumns.map((col) => (
                  <option key={col} value={col}>{col}</option>
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
                placeholder="např. memorability_score"
              />
            )}
          </div>

          <div className="flex justify-center mt-6">
            <Button
              onClick={() => onTrain(targetColumn)}
              disabled={!targetColumn || isTraining}
              title={!targetColumn ? "Vyberte sloupec s cílovou proměnnou" : undefined}
            >
              {isTraining ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Trénuji model... ({trainSecs}s)
                </>
              ) : (
                <>
                  <PlayCircle className="mr-2 h-4 w-4" /> Spustit trénink (RuleKit)
                </>
              )}
            </Button>
          </div>

          {/* Train results */}
          {trainResult && trainResult.status === "success" && (
            <div className={`rounded-xl p-4 space-y-3 border ${cls(deluxe, "bg-green-50 border-green-100", "bg-green-900/20 border-green-800/50")}`}>
              <div className={`flex items-center gap-2 ${cls(deluxe, "text-green-800", "text-green-400")}`}>
                <CheckCircle2 className="h-4 w-4" />
                <p className="text-sm font-bold">Trénink dokončen!</p>
              </div>
              <div className={`text-xs space-y-0.5 ${cls(deluxe, "text-green-700", "text-green-400/80")}`}>
                <p>Ensemble MSE: <strong>{trainResult.mse != null ? Number(trainResult.mse).toFixed(4) : "—"}</strong></p>
                {trainResult.rulekit_mse != null && (
                  <p>RuleKit MSE: <strong>{Number(trainResult.rulekit_mse).toFixed(4)}</strong></p>
                )}
                {trainResult.xgb_mse != null && (
                  <p>XGBoost MSE: <strong>{Number(trainResult.xgb_mse).toFixed(4)}</strong></p>
                )}
                {trainResult.cv_mse != null && (
                  <p>
                    Cross-val MSE ({trainResult.cv_folds ?? 5}-fold):{" "}
                    <strong>{Number(trainResult.cv_mse).toFixed(4)}</strong>
                    {trainResult.cv_std != null && (
                      <span className="opacity-70"> ± {Number(trainResult.cv_std).toFixed(4)}</span>
                    )}
                    {" "}
                    <span className={trainResult.cv_mse < (trainResult.mse ?? 0) * 2 ? "text-green-500" : "text-amber-500"}>
                      {trainResult.cv_mse < (trainResult.mse ?? 0) * 2 ? "✓ model zobecňuje" : "⚠ možné přetrénování"}
                    </span>
                  </p>
                )}
                {trainResult.warnings?.map((w, i) => (
                  <p key={i} className="text-amber-600">⚠ {w}</p>
                ))}
                <p>Vygenerováno pravidel: <strong>{trainResult.rules_count}</strong></p>
              </div>

              {/* Feature importance (top 5, collapsible) */}
              {trainResult.feature_importance?.xgboost && Object.keys(trainResult.feature_importance.xgboost).length > 0 && (
                <details className={`text-xs rounded border p-2 ${cls(deluxe, "border-slate-200", "border-slate-700")}`}>
                  <summary className={`cursor-pointer font-medium ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
                    Důležitost features (XGBoost — top 5)
                  </summary>
                  <ul className="mt-1.5 space-y-1">
                    {Object.entries(trainResult.feature_importance.xgboost)
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 5)
                      .map(([feat, score]) => (
                        <li key={feat} className={`flex items-center gap-2 ${cls(deluxe, "text-slate-600", "text-slate-400")}`}>
                          <span className="font-mono flex-1">{feat}</span>
                          <div className={`h-1.5 rounded-full bg-blue-400`} style={{ width: `${Math.round(score * 100)}px`, minWidth: "4px", maxWidth: "120px" }} />
                          <span className="w-10 text-right">{(score * 100).toFixed(1)}%</span>
                        </li>
                      ))}
                  </ul>
                </details>
              )}

              {trainResult.rules && trainResult.rules.length > 0 && (
                <div className={`p-3 rounded border text-left ${cls(deluxe, "bg-slate-100 border-slate-300", "bg-slate-800 border-slate-700")}`}>
                  <p className="font-bold mb-2 text-xs">Pravidlový model:</p>
                  <div className={`max-h-40 overflow-y-auto text-xs font-mono space-y-1 ${cls(deluxe, "text-slate-600", "text-slate-300")}`}>
                    {trainResult.rules.map((rule, idx) => (
                      <div key={idx} className="whitespace-pre-wrap break-words">
                        {idx + 1}. {rule}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Downloads */}
              <div className="grid grid-cols-2 gap-2 mt-3">
                <button
                  onClick={() => { if (trainResult.feature_spec) downloadFeatureSpec(trainResult.feature_spec); }}
                  className="px-2 py-1 bg-zinc-600 text-white rounded text-xs hover:bg-zinc-700 flex items-center justify-center gap-1"
                >
                  <Download className="w-3 h-3" /> Feature Spec
                </button>
                <button
                  onClick={() => { if (trainResult.training_data_X) downloadTrainingDataCSV(trainResult.training_data_X); }}
                  className="px-2 py-1 bg-zinc-600 text-white rounded text-xs hover:bg-zinc-700 flex items-center justify-center gap-1"
                >
                  <Download className="w-3 h-3" /> Training Data (X)
                </button>
                <button
                  onClick={() => { if (trainResult.rules) downloadRulesModel(trainResult.rules, trainResult.mse); }}
                  className="col-span-2 px-2 py-1 bg-zinc-600 text-white rounded text-xs hover:bg-zinc-700 flex items-center justify-center gap-1"
                >
                  <Download className="w-3 h-3" /> Pravidlový model
                </button>
              </div>

              {/* Continue button */}
              {onGoToStep && (
                <div className="flex justify-center mt-4">
                  <Button onClick={() => onGoToStep(4)}>
                    <ChevronRight className="mr-2 h-4 w-4" /> Pokračovat na Fázi 4
                  </Button>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ================================================================ */}
      {/*  FÁZE 4: TEST DATA FEATURE EXTRACTION                            */}
      {/* ================================================================ */}
      {step === 4 && (
        <div className="space-y-4">
          {featureSpec && (
            <FeatureSpecBox
              deluxe={deluxe}
              featureSpec={featureSpec}
              targetVariable={targetVariable}
            />
          )}

          {/* Toggle: upload vs server path */}
          <div className={`p-3 rounded-lg border ${cls(deluxe, "bg-slate-50 border-slate-200", "bg-slate-800/50 border-slate-700")}`}>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useServerPathTest}
                onChange={(e) => setUseServerPathTest(e.target.checked)}
                className="rounded"
              />
              <span className={`text-sm font-medium ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
                Soubor je již na serveru (velký dataset)
              </span>
            </label>
          </div>

          {!useServerPathTest ? (
            <FileDropZone
              deluxe={deluxe}
              file={testZipFile}
              onFile={setTestZipFile}
              accept=".zip"
              inputId="test-zip-upload"
              label="Nahrajte testovací ZIP dataset (jiná média než trénovací)"
              pickLabel="Vybrat testovací ZIP"
            />
          ) : (
            <div className="space-y-2">
              <label className={`text-sm font-medium ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
                Cesta k ZIP souboru na serveru
              </label>
              <input
                type="text"
                value={serverPathTest}
                onChange={(e) => setServerPathTest(e.target.value)}
                placeholder="/home/kovm23/test.zip"
                className={`w-full px-3 py-2 rounded border text-sm font-mono ${cls(deluxe, "bg-white border-slate-300 text-slate-800", "bg-slate-900 border-slate-600 text-slate-200")}`}
              />
            </div>
          )}

          {ollamaOk === false && (
            <div className={`flex items-start gap-2 p-3 rounded-lg border ${cls(deluxe, "bg-amber-50 border-amber-300 text-amber-800", "bg-amber-900/30 border-amber-700 text-amber-300")}`}>
              <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <p className="text-sm">Ollama není dostupný (localhost:11434). Spusťte: <code className="font-mono bg-black/10 px-1 rounded">ollama serve</code></p>
              </div>
              {recheckOllama && (
                <button onClick={recheckOllama} className="text-xs underline opacity-70 hover:opacity-100 whitespace-nowrap">Zkontrolovat znovu</button>
              )}
            </div>
          )}

          <div className="flex justify-center mt-6">
            <Button
              onClick={() => {
                if (useServerPathTest) {
                  onExtractTestingLocal(serverPathTest);
                } else {
                  testZipFile && onExtractTesting(testZipFile);
                }
              }}
              disabled={(useServerPathTest ? !serverPathTest : !testZipFile) || isExtractingTest}
              title={!useServerPathTest && !testZipFile ? "Nejdříve nahrajte testovací ZIP" : undefined}
            >
              {isExtractingTest ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Extrahuji features...
                </>
              ) : (
                <>
                  <PlayCircle className="mr-2 h-4 w-4" /> Spustit testovací extrakci
                </>
              )}
            </Button>
          </div>

          {isExtractingTest && (
            <div className="space-y-2">
              <ProgressBar deluxe={deluxe} progress={progress} label={progressLabel} />
              {testExtractStalled && (
                <p className={`text-xs ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>
                  ℹ Zpracování probíhá — větší soubory nebo pomalý model mohou trvat i 10+ minut.
                </p>
              )}
              <div className="flex gap-2">
                {onCancel && (
                  <Button variant="outline" size="sm" onClick={onCancel} className="text-xs">
                    ✕ Zastavit
                  </Button>
                )}
              </div>
            </div>
          )}

          {/* Výsledek extrakce + pokračovat */}
          {testingDataX && !isExtractingTest && (
            <div className={`flex items-center gap-2 p-2 rounded-lg ${cls(deluxe, "bg-green-50 text-green-700", "bg-green-900/30 text-green-400")}`}>
              <CheckCircle2 className="h-4 w-4" />
              <span className="text-sm font-medium">Testovací extrakce dokončena — {testingDataX.length} řádků</span>
            </div>
          )}
          {testingDataX && (
            <>
              <DatasetTable deluxe={deluxe} data={testingDataX} title="Testing Dataset X" />
              <div className="flex justify-center mt-4">
                <Button onClick={() => onGoToStep?.(5)}>
                  <ChevronRight className="mr-2 h-4 w-4" /> Pokračovat na Fázi 5
                </Button>
              </div>
            </>
          )}
        </div>
      )}

      {/* ================================================================ */}
      {/*  FÁZE 5: PREDIKCE                                                */}
      {/* ================================================================ */}
      {step === 5 && (
        <div className="space-y-4">
          {/* Testing dataset_X preview */}
          {testingDataX && (
            <DatasetTable deluxe={deluxe} data={testingDataX} title="Testing Dataset X" />
          )}

          {/* Optional testing_Y labels */}
          {!predictions && (
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
                  Nahrát testing dataset_Y (pro vyhodnocení přesnosti)
                </span>
              </label>
              <p className={`text-xs mt-1 ${cls(deluxe, "text-amber-600", "text-amber-400/70")}`}>
                CSV se skutečnými hodnotami cílové proměnné. Slouží k porovnání predikcí vs. realita.
              </p>
              {useTestingLabels && (
                <div className="mt-2">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => { if (e.target.files?.[0]) setTestingLabels(e.target.files[0]); }}
                    className={`text-xs ${cls(deluxe, "text-slate-600", "text-slate-400")}`}
                  />
                  {testingLabels && (
                    <p className="mt-1 text-xs text-green-500 font-medium">CSV: {testingLabels.name}</p>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Predict button (if predictions not yet available) */}
          {!predictions && (
            <div className="flex justify-center mt-6">
              <Button
                onClick={() => onPredict(useTestingLabels ? testingLabels : null)}
                disabled={isPredicting}
              >
                {isPredicting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Predikuji... ({predictSecs}s)
                  </>
                ) : (
                  <>
                    <PlayCircle className="mr-2 h-4 w-4" /> Spustit predikci
                  </>
                )}
              </Button>
            </div>
          )}

          {/* Predictions table */}
          {predictions && predictions.length > 0 && (
            <div className={`rounded-xl p-4 space-y-3 border ${cls(deluxe, "bg-green-50 border-green-100", "bg-green-900/20 border-green-800/50")}`}>
              <div className={`flex items-center gap-2 ${cls(deluxe, "text-green-800", "text-green-400")}`}>
                <CheckCircle2 className="h-4 w-4" />
                <p className="text-sm font-bold">
                  Predikce dokončena ({predictions.length} objektů)
                </p>
              </div>

              {/* Metrics panel */}
              {predictionMetrics && (
                <div className={`p-3 rounded-lg border ${cls(deluxe, "bg-blue-50 border-blue-200", "bg-blue-900/20 border-blue-800/50")}`}>
                  <p className={`text-xs font-bold mb-2 ${cls(deluxe, "text-blue-900", "text-blue-300")}`}>
                    Vyhodnocení modelu (predikce vs. skutečnost):
                  </p>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {(() => {
                      const mse = typeof predictionMetrics.mse === "number" ? predictionMetrics.mse : 0;
                      const mae = typeof predictionMetrics.mae === "number" ? predictionMetrics.mae : 0;
                      const corr = predictionMetrics.correlation;
                      const matchPct = predictionMetrics.total_count > 0
                        ? Math.round((predictionMetrics.matched_count / predictionMetrics.total_count) * 100)
                        : 0;
                      // Color based on quality: lower MSE/MAE = better (green), higher corr = better
                      const corrColor = corr !== null && corr > 0.5 ? "text-green-600" : corr !== null && corr > 0 ? "text-yellow-600" : "text-red-500";
                      return (
                        <>
                          <div className="text-center">
                            <p className={`text-lg font-bold ${cls(deluxe, "text-blue-700", "text-blue-400")}`}>
                              {typeof mse === "number" ? mse.toFixed(4) : mse}
                            </p>
                            <p className={`text-[10px] ${cls(deluxe, "text-blue-600", "text-blue-500")}`}>MSE (nižší = lepší)</p>
                          </div>
                          <div className="text-center">
                            <p className={`text-lg font-bold ${cls(deluxe, "text-blue-700", "text-blue-400")}`}>
                              {typeof mae === "number" ? mae.toFixed(4) : mae}
                            </p>
                            <p className={`text-[10px] ${cls(deluxe, "text-blue-600", "text-blue-500")}`}>MAE (nižší = lepší)</p>
                          </div>
                          <div className="text-center">
                            <p className={`text-lg font-bold ${corrColor}`}>
                              {corr !== null ? (typeof corr === "number" ? corr.toFixed(4) : corr) : "N/A"}
                            </p>
                            <p className={`text-[10px] ${cls(deluxe, "text-blue-600", "text-blue-500")}`}>Korelace (vyšší = lepší)</p>
                          </div>
                          <div className="text-center">
                            <p className={`text-lg font-bold ${matchPct === 100 ? "text-green-600" : matchPct > 50 ? "text-yellow-600" : "text-red-500"}`}>
                              {matchPct}%
                            </p>
                            <p className={`text-[10px] ${cls(deluxe, "text-blue-600", "text-blue-500")}`}>
                              Spárováno ({predictionMetrics.matched_count}/{predictionMetrics.total_count})
                            </p>
                          </div>
                        </>
                      );
                    })()}
                  </div>
                </div>
              )}

              {/* Predictions table */}
              <div className={`rounded-lg border overflow-hidden ${cls(deluxe, "border-slate-200", "border-slate-700")}`}>
                <div className="overflow-x-auto max-h-80">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className={cls(deluxe, "bg-slate-50", "bg-slate-900")}>
                        <th className="px-2 py-1 text-left font-mono">media_name</th>
                        <th className="px-2 py-1 text-left font-mono">predicted_score</th>
                        {predictions.some((p) => p.actual_score !== undefined) && (
                          <th className="px-2 py-1 text-left font-mono">actual_score</th>
                        )}
                        <th className="px-2 py-1 text-left font-mono">rule_applied</th>
                        {featureSpec && Object.keys(featureSpec).map((f) => (
                          <th key={f} className="px-2 py-1 text-left font-mono">{f}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.map((pred, i) => (
                        <tr key={i} className={i % 2 === 0 ? cls(deluxe, "bg-white", "bg-slate-800/50") : cls(deluxe, "bg-slate-50/50", "bg-slate-900/50")}>
                          <td className="px-2 py-1 whitespace-nowrap font-medium">{pred.media_name}</td>
                          <td className="px-2 py-1 whitespace-nowrap font-bold text-blue-600">{pred.predicted_score}</td>
                          {predictions.some((p) => p.actual_score !== undefined) && (
                            <td className="px-2 py-1 whitespace-nowrap font-bold text-green-600">
                              {pred.actual_score !== undefined ? pred.actual_score : "—"}
                            </td>
                          )}
                          <td className="px-2 py-1 whitespace-nowrap font-mono text-[10px]">{pred.rule_applied}</td>
                          {featureSpec && Object.keys(featureSpec).map((f) => (
                            <td key={f} className="px-2 py-1 whitespace-nowrap">
                              {String(pred.extracted_features[f] ?? "")}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Download */}
              <div className="flex justify-center gap-2 flex-wrap">
                <button
                  onClick={() => {
                    const hasActual = predictions.some((p) => p.actual_score !== undefined);
                    const headers = ["media_name", "predicted_score", ...(hasActual ? ["actual_score"] : []), "rule_applied", ...(featureSpec ? Object.keys(featureSpec) : [])];
                    const csv = [
                      headers.join(","),
                      ...predictions.map((p) => [
                        p.media_name,
                        p.predicted_score,
                        ...(hasActual ? [p.actual_score !== undefined ? p.actual_score : ""] : []),
                        `"${p.rule_applied}"`,
                        ...(featureSpec ? Object.keys(featureSpec).map((f) => String(p.extracted_features[f] ?? "")) : []),
                      ].join(","))
                    ].join("\n");
                    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `predictions_${new Date().toISOString().slice(0, 10)}.csv`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="px-3 py-1.5 bg-slate-500 text-white rounded text-xs hover:bg-slate-600 flex items-center gap-1"
                >
                  <Download className="w-3 h-3" /> Predikce (CSV)
                </button>
                <button
                  onClick={() => downloadExperimentZip({
                    featureSpec: featureSpec ?? undefined,
                    trainingDataX: trainResult?.training_data_X ?? undefined,
                    testingDataX: testingDataX ?? undefined,
                    rules: trainResult?.rules ?? undefined,
                    mse: trainResult?.mse ?? undefined,
                    predictions: predictions as unknown as Record<string, unknown>[],
                    metrics: predictionMetrics as unknown as Record<string, unknown> ?? undefined,
                  })}
                  className="px-3 py-1.5 bg-slate-700 text-white rounded text-xs hover:bg-slate-800 flex items-center gap-1"
                >
                  <Download className="w-3 h-3" /> Stáhnout experiment (ZIP)
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
