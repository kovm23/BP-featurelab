import React, { useEffect, useRef, useState } from "react";
import { AlertTriangle, Pencil, Plus, Trash2, UploadCloud, X } from "lucide-react";
import type { FeatureSpec, FeatureSpecValue, PredictionMetrics } from "@/lib/api";
import type { TrainingTranslations } from "./translations";

export function cls(deluxe: boolean, light: string, dark: string) {
  return deluxe ? dark : light;
}

export function isClassificationMetrics(metrics: PredictionMetrics | null): boolean {
  if (!metrics) return false;
  return metrics.mode === "classification" || typeof metrics.accuracy === "number";
}

export function FileDropZone({
  deluxe,
  uiLanguage = "cs",
  file,
  onFile,
  accept,
  inputId,
  label,
  pickLabel,
  selectedLabel,
}: {
  deluxe: boolean;
  uiLanguage?: "cs" | "en";
  file: File | null;
  onFile: (f: File) => void;
  accept: string;
  inputId: string;
  label: string;
  pickLabel: string;
  selectedLabel?: string;
}) {
  const selectedText = selectedLabel || (uiLanguage === "en" ? "Selected" : "Vybráno");
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
        className={`h-8 w-8 mx-auto mb-3 ${cls(deluxe, "text-slate-400", "text-slate-500")}`}
      />
      <p
        className={`text-sm font-medium mb-1 ${cls(deluxe, "text-slate-700", "text-slate-300")}`}
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
          {selectedText}: {file.name}
        </p>
      )}
    </div>
  );
}

export function ProgressBar({
  deluxe,
  progress,
  label,
  etaText,
}: {
  deluxe: boolean;
  progress: number;
  label: string;
  etaText?: string | null;
}) {
  const inProgress = progress > 0 && progress < 100;
  const isDone = progress >= 100;
  const fillColor = isDone ? "bg-emerald-500" : "bg-blue-500";

  return (
    <div className="mt-6 space-y-1.5">
      <div className="flex justify-between items-center text-xs font-medium gap-2">
        <span
          className={`truncate ${cls(deluxe, "text-slate-600", "text-slate-300")}`}
          title={label}
        >
          {label}
        </span>
        <span className={`shrink-0 tabular-nums ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>
          {progress}%
        </span>
      </div>
      <div
        className={`h-3 rounded-full overflow-hidden ${cls(deluxe, "bg-slate-100", "bg-slate-800/60")}`}
      >
        <div
          className={`h-full rounded-full transition-all duration-700 relative overflow-hidden ${fillColor}`}
          style={{ width: `${Math.max(inProgress ? 3 : 0, progress)}%` }}
        >
          {inProgress && (
            <div
              className="absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-white/30 to-transparent"
              style={{ animation: "progress-sweep 1.8s ease-in-out infinite" }}
            />
          )}
        </div>
      </div>
      {etaText && (
        <p className={`text-[11px] ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>
          {etaText}
        </p>
      )}
    </div>
  );
}

export function FeatureSpecBox({
  deluxe,
  uiLanguage = "cs",
  featureSpec,
  targetVariable,
  editable = false,
  onUpdate,
}: {
  deluxe: boolean;
  uiLanguage?: "cs" | "en";
  featureSpec: FeatureSpec;
  targetVariable: string;
  editable?: boolean;
  onUpdate?: (spec: FeatureSpec) => void;
}) {
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editMode, setEditMode] = useState<"range" | "categories">("range");
  const [editMin, setEditMin] = useState(0);
  const [editMax, setEditMax] = useState(10);
  const [editCategories, setEditCategories] = useState("");
  const [addingNew, setAddingNew] = useState(false);
  const [newName, setNewName] = useState("");
  const [newMode, setNewMode] = useState<"range" | "categories">("range");
  const [newMin, setNewMin] = useState(0);
  const [newMax, setNewMax] = useState(10);
  const [newCategories, setNewCategories] = useState("");

  const tx = uiLanguage === "en"
    ? {
        title: "Feature Definition Spec",
        target: "Target",
        editable: "Editable",
        range: "Range",
        categories: "Categories",
        featureName: "Feature name",
        newFeatureName: "New feature name (e.g. scene_brightness)",
        save: "Save",
        cancel: "Cancel",
        edit: "Edit",
        remove: "Remove",
        addFeature: "Add feature",
        add: "Add",
        rangeSuffix: "range",
      }
    : {
        title: "Feature Definition Spec",
        target: "Cíl",
        editable: "Editovatelné",
        range: "Rozsah",
        categories: "Kategorie",
        featureName: "Název feature",
        newFeatureName: "Název nové feature (např. scene_brightness)",
        save: "Uložit",
        cancel: "Zrušit",
        edit: "Upravit",
        remove: "Odebrat",
        addFeature: "Přidat feature",
        add: "Přidat",
        rangeSuffix: "rozsah",
      };

  const formatSpecValue = (value: FeatureSpecValue | string): string => {
    if (Array.isArray(value)) {
      if (value.length === 2 && value.every((v) => typeof v === "number")) {
        return `${value[0]}–${value[1]} (${tx.rangeSuffix})`;
      }
      return `[${value.map((v) => JSON.stringify(v)).join(", ")}]`;
    }
    return String(value);
  };

  const parseEditorState = (value: FeatureSpecValue | string) => {
    if (Array.isArray(value)) {
      if (value.length === 2 && value.every((v) => typeof v === "number")) {
        setEditMode("range");
        setEditMin(Number(value[0]));
        setEditMax(Number(value[1]));
        return;
      }
      if (value.every((v) => typeof v === "string")) {
        setEditMode("categories");
        setEditCategories(value.join(", "));
        return;
      }
    }
    setEditMode("categories");
    setEditCategories(String(value));
  };

  const buildEditedValue = (): FeatureSpecValue => {
    if (editMode === "range") {
      return [Number(editMin), Number(editMax)];
    }
    return editCategories
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  };

  const buildNewValue = (): FeatureSpecValue => {
    if (newMode === "range") {
      return [Number(newMin), Number(newMax)];
    }
    return newCategories
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  };

  const startEdit = (key: string, value: FeatureSpecValue) => {
    setEditingKey(key);
    setEditName(key);
    parseEditorState(value);
  };

  const saveEdit = () => {
    if (!editingKey || !editName.trim() || !onUpdate) return;
    const updated = { ...featureSpec };
    if (editName !== editingKey) delete updated[editingKey];
    updated[editName.trim()] = buildEditedValue();
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
    onUpdate({ ...featureSpec, [newName.trim()]: buildNewValue() });
    setNewName("");
    setNewMode("range");
    setNewMin(0);
    setNewMax(10);
    setNewCategories("");
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
          className={`text-sm font-bold ${cls(deluxe, "text-blue-900", "text-white")}`}
        >
          {tx.title} ({tx.target}: {targetVariable}):
        </h3>
        {editable && (
          <span
            className={`text-[10px] px-1.5 py-0.5 rounded ${cls(
              deluxe,
              "bg-amber-100 text-amber-700",
              "bg-amber-900/40 text-amber-400"
            )}`}
          >
            {tx.editable}
          </span>
        )}
      </div>
      <ul className={`text-xs space-y-1.5 ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
        {Object.entries(featureSpec).map(([key, desc]) =>
          editable && editingKey === key ? (
            <li key={key} className="flex flex-col gap-1 p-2 rounded bg-black/5">
              <input
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                className={`text-xs font-mono px-1.5 py-0.5 rounded border outline-none ${cls(
                  deluxe,
                  "bg-white border-slate-300",
                  "bg-slate-800 border-slate-600 text-white"
                )}`}
                placeholder={tx.featureName}
              />
              <div className="flex gap-1 text-[10px]">
                <button
                  type="button"
                  onClick={() => setEditMode("range")}
                  className={`px-1.5 py-0.5 rounded ${editMode === "range" ? "bg-blue-500 text-white" : "bg-black/10"}`}
                >
                  {tx.range}
                </button>
                <button
                  type="button"
                  onClick={() => setEditMode("categories")}
                  className={`px-1.5 py-0.5 rounded ${editMode === "categories" ? "bg-blue-500 text-white" : "bg-black/10"}`}
                >
                  {tx.categories}
                </button>
              </div>
              {editMode === "range" && (
                <div className="grid grid-cols-2 gap-1">
                  <input
                    type="number"
                    value={editMin}
                    onChange={(e) => setEditMin(Number(e.target.value))}
                    className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(
                      deluxe,
                      "bg-white border-slate-300",
                      "bg-slate-800 border-slate-600 text-white"
                    )}`}
                    placeholder="min"
                  />
                  <input
                    type="number"
                    value={editMax}
                    onChange={(e) => setEditMax(Number(e.target.value))}
                    className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(
                      deluxe,
                      "bg-white border-slate-300",
                      "bg-slate-800 border-slate-600 text-white"
                    )}`}
                    placeholder="max"
                  />
                </div>
              )}
              {editMode === "categories" && (
                <input
                  value={editCategories}
                  onChange={(e) => setEditCategories(e.target.value)}
                  className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(
                    deluxe,
                    "bg-white border-slate-300",
                    "bg-slate-800 border-slate-600 text-white"
                  )}`}
                  placeholder="cat_a, cat_b, cat_c"
                />
              )}
              <div className="flex gap-1">
                <button
                  onClick={saveEdit}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-green-500 text-white hover:bg-green-600"
                >
                  {tx.save}
                </button>
                <button
                  onClick={() => setEditingKey(null)}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-slate-400 text-white hover:bg-slate-500"
                >
                  {tx.cancel}
                </button>
              </div>
            </li>
          ) : (
            <li key={key} className="flex items-center gap-2 group">
              <span className="font-mono bg-black/10 px-1 rounded">{key}</span>:{" "}
              <span className="flex-1">{formatSpecValue(desc)}</span>
              {editable && (
                <span className="flex gap-0.5 opacity-60 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => startEdit(key, desc)}
                    title={tx.edit}
                    className="p-0.5 rounded hover:bg-black/10"
                  >
                    <Pencil className="h-3 w-3" />
                  </button>
                  <button
                    onClick={() => removeFeature(key)}
                    title={tx.remove}
                    className="p-0.5 rounded hover:bg-red-100 text-red-500"
                  >
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
                className={`text-xs font-mono px-1.5 py-0.5 rounded border outline-none ${cls(
                  deluxe,
                  "bg-white border-slate-300",
                  "bg-slate-800 border-slate-600 text-white"
                )}`}
                placeholder={tx.newFeatureName}
              />
              <div className="flex gap-1 text-[10px]">
                <button
                  type="button"
                  onClick={() => setNewMode("range")}
                  className={`px-1.5 py-0.5 rounded ${newMode === "range" ? "bg-blue-500 text-white" : "bg-black/10"}`}
                >
                  {tx.range}
                </button>
                <button
                  type="button"
                  onClick={() => setNewMode("categories")}
                  className={`px-1.5 py-0.5 rounded ${newMode === "categories" ? "bg-blue-500 text-white" : "bg-black/10"}`}
                >
                  {tx.categories}
                </button>
              </div>
              {newMode === "range" ? (
                <div className="grid grid-cols-2 gap-1">
                  <input
                    type="number"
                    value={newMin}
                    onChange={(e) => setNewMin(Number(e.target.value))}
                    className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(
                      deluxe,
                      "bg-white border-slate-300",
                      "bg-slate-800 border-slate-600 text-white"
                    )}`}
                    placeholder="min"
                  />
                  <input
                    type="number"
                    value={newMax}
                    onChange={(e) => setNewMax(Number(e.target.value))}
                    className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(
                      deluxe,
                      "bg-white border-slate-300",
                      "bg-slate-800 border-slate-600 text-white"
                    )}`}
                    placeholder="max"
                  />
                </div>
              ) : (
                <input
                  value={newCategories}
                  onChange={(e) => setNewCategories(e.target.value)}
                  className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(
                    deluxe,
                    "bg-white border-slate-300",
                    "bg-slate-800 border-slate-600 text-white"
                  )}`}
                  placeholder="cat_a, cat_b, cat_c"
                />
              )}
              <div className="flex gap-1">
                <button
                  onClick={addFeature}
                  disabled={!newName.trim()}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-green-500 text-white hover:bg-green-600 disabled:opacity-50"
                >
                  {tx.add}
                </button>
                <button
                  onClick={() => setAddingNew(false)}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-slate-400 text-white hover:bg-slate-500"
                >
                  {tx.cancel}
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setAddingNew(true)}
              className={`flex items-center gap-1 text-[10px] mt-1 px-2 py-0.5 rounded hover:bg-black/10 ${cls(
                deluxe,
                "text-blue-600",
                "text-blue-400"
              )}`}
            >
              <Plus className="h-3 w-3" /> {tx.addFeature}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export function DatasetTable({
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
                <th
                  key={c}
                  className={`px-2 py-1 text-left font-mono whitespace-nowrap ${cls(
                    deluxe,
                    "text-slate-600",
                    "text-slate-400"
                  )}`}
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr
                key={i}
                className={i % 2 === 0 ? cls(deluxe, "bg-white", "bg-slate-800/50") : cls(deluxe, "bg-slate-50/50", "bg-slate-900/50")}
              >
                {cols.map((c) => (
                  <td
                    key={c}
                    className={`px-2 py-1 whitespace-nowrap ${cls(deluxe, "text-slate-700", "text-slate-300")}`}
                  >
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

export function useElapsedTimer(active: boolean) {
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    if (!active) {
      setSecs(0);
      return;
    }
    setSecs(0);
    const id = setInterval(() => setSecs((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [active]);
  return secs;
}

export function formatDurationShort(totalSeconds: number) {
  const seconds = Math.max(0, Math.round(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

export function useEstimatedRemaining(progress: number, active: boolean) {
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);
  const samplesRef = useRef<Array<{ time: number; progress: number }>>([]);
  const lastProgressRef = useRef<number | null>(null);

  const reset = () => {
    samplesRef.current = [];
    lastProgressRef.current = null;
    setEtaSeconds(null);
  };

  useEffect(() => {
    if (!active) {
      reset();
    } else {
      const now = Date.now();
      samplesRef.current = [{ time: now, progress }];
      lastProgressRef.current = progress;
    }
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!active) return;
    if (progress < 8 || progress >= 99) {
      if (progress >= 99) setEtaSeconds(null);
      lastProgressRef.current = progress;
      return;
    }

    const now = Date.now();
    const lastProgress = lastProgressRef.current;

    if (lastProgress !== null && progress < lastProgress) {
      samplesRef.current = [{ time: now, progress }];
      setEtaSeconds(null);
      lastProgressRef.current = progress;
      return;
    }

    if (lastProgress === progress) return;

    samplesRef.current = [...samplesRef.current, { time: now, progress }]
      .filter((sample) => now - sample.time <= 90_000)
      .slice(-8);
    lastProgressRef.current = progress;

    const first = samplesRef.current[0];
    const last = samplesRef.current[samplesRef.current.length - 1];
    if (!first || !last) return;

    const deltaProgress = last.progress - first.progress;
    const deltaSeconds = (last.time - first.time) / 1000;
    if (deltaProgress <= 0 || deltaSeconds < 5) return;

    const speedPerSecond = deltaProgress / deltaSeconds;
    const rawEta = (100 - progress) / speedPerSecond;
    if (!Number.isFinite(rawEta) || rawEta <= 0) return;

    const rounded = Math.max(1, Math.round(rawEta));
    setEtaSeconds((prev) => (prev == null ? rounded : Math.min(prev, rounded)));
  }, [progress, active]);

  useEffect(() => {
    if (!active) return;
    const intervalId = setInterval(() => {
      setEtaSeconds((prev) => {
        if (prev == null) return null;
        return prev > 1 ? prev - 1 : 1;
      });
    }, 1000);
    return () => clearInterval(intervalId);
  }, [active]);

  return etaSeconds;
}

export function useProgressStall(progress: number, active: boolean, thresholdMs = 90_000) {
  const [stalled, setStalled] = useState(false);
  const lastRef = useRef({ val: -1, time: 0 });

  useEffect(() => {
    if (!active) {
      setStalled(false);
      return;
    }
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

export function enrichError(raw: string): { message: string; hint?: string } {
  if (raw.includes("No data remained after joining")) {
    return {
      message: raw,
      hint:
        'Zkontroluj, že názvy souborů v CSV (první sloupec bez přípony) odpovídají názvům médií v ZIPu. Např. soubor "video.mp4" → řádek CSV musí mít hodnotu "video".',
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
    return {
      message: raw,
      hint: "Vrať se na Fázi 2 a dokonči extrakci trénovacích dat.",
    };
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

export function QueueBusyBanner({
  deluxe,
  queuedCount = 0,
  tr,
}: {
  deluxe: boolean;
  queuedCount?: number;
  tr: TrainingTranslations;
}) {
  if (queuedCount <= 0) return null;
  return (
    <div
      className={`mb-4 flex items-start gap-2 p-3 rounded-lg border ${cls(
        deluxe,
        "bg-blue-50 border-blue-200 text-blue-800",
        "bg-blue-900/30 border-blue-800/50 text-blue-300"
      )}`}
    >
      <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
      <p className="text-sm">{tr.queueBusyMessage.replace("{count}", String(queuedCount))}</p>
    </div>
  );
}

