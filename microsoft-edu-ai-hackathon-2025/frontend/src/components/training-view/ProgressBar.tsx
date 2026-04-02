import { cls } from "./style";

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
        <span className={`truncate ${cls(deluxe, "text-slate-600", "text-slate-300")}`} title={label}>
          {label}
        </span>
        <span className={`shrink-0 tabular-nums ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>
          {progress}%
        </span>
      </div>
      <div className={`h-3 rounded-full overflow-hidden ${cls(deluxe, "bg-slate-100", "bg-slate-800/60")}`}>
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
      {etaText && <p className={`text-[11px] ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>{etaText}</p>}
    </div>
  );
}
