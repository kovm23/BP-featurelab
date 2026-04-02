import { AlertTriangle } from "lucide-react";
import { cls } from "./shared";
import type { TrainingTranslations } from "./translations";

export function OllamaWarning({
  deluxe,
  tr,
  recheckOllama,
}: {
  deluxe: boolean;
  tr: TrainingTranslations;
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
