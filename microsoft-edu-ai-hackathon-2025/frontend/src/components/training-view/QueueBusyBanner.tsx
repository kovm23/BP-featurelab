import { AlertTriangle } from "lucide-react";
import { cls } from "./style";
import type { TrainingTranslations } from "./translations";

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
