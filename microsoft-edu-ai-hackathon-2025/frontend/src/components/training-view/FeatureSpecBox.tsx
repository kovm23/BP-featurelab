import { useState } from "react";
import { Pencil, Plus, Trash2 } from "lucide-react";
import type { FeatureSpec, FeatureSpecValue } from "@/lib/api";
import { cls } from "./style";

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
        title: "Specifikace featur",
        target: "Cíl",
        editable: "Editovatelné",
        range: "Rozsah",
        categories: "Kategorie",
        featureName: "Název featury",
        newFeatureName: "Název nové featury (např. scene_brightness)",
        save: "Uložit",
        cancel: "Zrušit",
        edit: "Upravit",
        remove: "Odebrat",
        addFeature: "Přidat featuru",
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
    if (editMode === "range") return [Number(editMin), Number(editMax)];
    return editCategories.split(",").map((s) => s.trim()).filter(Boolean);
  };

  const buildNewValue = (): FeatureSpecValue => {
    if (newMode === "range") return [Number(newMin), Number(newMax)];
    return newCategories.split(",").map((s) => s.trim()).filter(Boolean);
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
    <div className={`p-4 rounded-lg border ${cls(deluxe, "bg-blue-50 border-blue-100", "bg-slate-900 border-slate-700")}`}>
      <div className="flex items-center justify-between mb-2">
        <h3 className={`text-sm font-bold ${cls(deluxe, "text-blue-900", "text-white")}`}>
          {tx.title} ({tx.target}: {targetVariable}):
        </h3>
        {editable && (
          <span className={`text-[10px] px-1.5 py-0.5 rounded ${cls(deluxe, "bg-amber-100 text-amber-700", "bg-amber-900/40 text-amber-400")}`}>
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
                className={`text-xs font-mono px-1.5 py-0.5 rounded border outline-none ${cls(deluxe, "bg-white border-slate-300", "bg-slate-800 border-slate-600 text-white")}`}
                placeholder={tx.featureName}
              />
              <div className="flex gap-1 text-[10px]">
                <button type="button" onClick={() => setEditMode("range")} className={`px-1.5 py-0.5 rounded ${editMode === "range" ? "bg-blue-500 text-white" : "bg-black/10"}`}>{tx.range}</button>
                <button type="button" onClick={() => setEditMode("categories")} className={`px-1.5 py-0.5 rounded ${editMode === "categories" ? "bg-blue-500 text-white" : "bg-black/10"}`}>{tx.categories}</button>
              </div>
              {editMode === "range" && (
                <div className="grid grid-cols-2 gap-1">
                  <input type="number" value={editMin} onChange={(e) => setEditMin(Number(e.target.value))} className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(deluxe, "bg-white border-slate-300", "bg-slate-800 border-slate-600 text-white")}`} placeholder="min" />
                  <input type="number" value={editMax} onChange={(e) => setEditMax(Number(e.target.value))} className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(deluxe, "bg-white border-slate-300", "bg-slate-800 border-slate-600 text-white")}`} placeholder="max" />
                </div>
              )}
              {editMode === "categories" && (
                <input
                  value={editCategories}
                  onChange={(e) => setEditCategories(e.target.value)}
                  className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(deluxe, "bg-white border-slate-300", "bg-slate-800 border-slate-600 text-white")}`}
                  placeholder="cat_a, cat_b, cat_c"
                />
              )}
              <div className="flex gap-1">
                <button onClick={saveEdit} className="text-[10px] px-1.5 py-0.5 rounded bg-green-500 text-white hover:bg-green-600">{tx.save}</button>
                <button onClick={() => setEditingKey(null)} className="text-[10px] px-1.5 py-0.5 rounded bg-slate-400 text-white hover:bg-slate-500">{tx.cancel}</button>
              </div>
            </li>
          ) : (
            <li key={key} className="flex items-center gap-2 group">
              <span className="font-mono bg-black/10 px-1 rounded">{key}</span>: <span className="flex-1">{formatSpecValue(desc)}</span>
              {editable && (
                <span className="flex gap-0.5 opacity-60 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
                  <button onClick={() => startEdit(key, desc)} title={tx.edit} className="p-0.5 rounded hover:bg-black/10"><Pencil className="h-3 w-3" /></button>
                  <button onClick={() => removeFeature(key)} title={tx.remove} className="p-0.5 rounded hover:bg-red-100 text-red-500"><Trash2 className="h-3 w-3" /></button>
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
                placeholder={tx.newFeatureName}
              />
              <div className="flex gap-1 text-[10px]">
                <button type="button" onClick={() => setNewMode("range")} className={`px-1.5 py-0.5 rounded ${newMode === "range" ? "bg-blue-500 text-white" : "bg-black/10"}`}>{tx.range}</button>
                <button type="button" onClick={() => setNewMode("categories")} className={`px-1.5 py-0.5 rounded ${newMode === "categories" ? "bg-blue-500 text-white" : "bg-black/10"}`}>{tx.categories}</button>
              </div>
              {newMode === "range" ? (
                <div className="grid grid-cols-2 gap-1">
                  <input type="number" value={newMin} onChange={(e) => setNewMin(Number(e.target.value))} className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(deluxe, "bg-white border-slate-300", "bg-slate-800 border-slate-600 text-white")}`} placeholder="min" />
                  <input type="number" value={newMax} onChange={(e) => setNewMax(Number(e.target.value))} className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(deluxe, "bg-white border-slate-300", "bg-slate-800 border-slate-600 text-white")}`} placeholder="max" />
                </div>
              ) : (
                <input
                  value={newCategories}
                  onChange={(e) => setNewCategories(e.target.value)}
                  className={`text-xs px-1.5 py-0.5 rounded border outline-none ${cls(deluxe, "bg-white border-slate-300", "bg-slate-800 border-slate-600 text-white")}`}
                  placeholder="cat_a, cat_b, cat_c"
                />
              )}
              <div className="flex gap-1">
                <button onClick={addFeature} disabled={!newName.trim()} className="text-[10px] px-1.5 py-0.5 rounded bg-green-500 text-white hover:bg-green-600 disabled:opacity-50">{tx.add}</button>
                <button onClick={() => setAddingNew(false)} className="text-[10px] px-1.5 py-0.5 rounded bg-slate-400 text-white hover:bg-slate-500">{tx.cancel}</button>
              </div>
            </div>
          ) : (
            <button onClick={() => setAddingNew(true)} className={`flex items-center gap-1 text-[10px] mt-1 px-2 py-0.5 rounded hover:bg-black/10 ${cls(deluxe, "text-blue-600", "text-blue-400")}`}>
              <Plus className="h-3 w-3" /> {tx.addFeature}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
