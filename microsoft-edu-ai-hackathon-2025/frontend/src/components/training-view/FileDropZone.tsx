import { UploadCloud } from "lucide-react";
import { cls } from "./style";

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
      <UploadCloud className={`h-8 w-8 mx-auto mb-3 ${cls(deluxe, "text-slate-400", "text-slate-500")}`} />
      <p className={`text-sm font-medium mb-1 ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>{label}</p>
      <input
        type="file"
        id={inputId}
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.[0]) onFile(e.target.files[0]);
        }}
        accept={accept}
      />
      <label htmlFor={inputId} className="cursor-pointer text-blue-500 hover:text-blue-600 text-sm font-medium">
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
