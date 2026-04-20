import { cls } from "./shared";

export function MetricStat({
  deluxe,
  value,
  label,
  valueClass,
  tooltip,
}: {
  deluxe: boolean;
  value: string;
  label: string;
  valueClass?: string;
  tooltip?: string;
}) {
  return (
    <div className="relative text-center group py-1 cursor-help">
      <p className={`text-xl font-semibold leading-none ${valueClass ?? cls(deluxe, "text-blue-700", "text-blue-400")}`}>{value}</p>
      <p className={`mt-1 text-[11px] leading-tight ${cls(deluxe, "text-blue-700/80", "text-blue-400/80")}`}>{label}</p>

      {tooltip && (
        <div className={`absolute left-1/2 bottom-full -translate-x-1/2 mb-2 w-52 px-2.5 py-2 rounded-md text-[11px] leading-snug pointer-events-none opacity-0 translate-y-1 transition-all duration-150 z-50 group-hover:opacity-100 group-hover:translate-y-0 ${cls(
          deluxe,
          "bg-slate-900 text-slate-50",
          "bg-slate-100 text-slate-900"
        )} shadow-md`}>
          {tooltip}
        </div>
      )}
    </div>
  );
}

export function ConfidenceStat({
  deluxe,
  label,
  value,
  valueClass,
}: {
  deluxe: boolean;
  label: string;
  value: number | null | undefined;
  valueClass?: string;
}) {
  return (
    <div className="text-center py-1">
      <p className={`text-[10px] ${cls(deluxe, "text-blue-700/70", "text-blue-400/70")}`}>{label}</p>
      <p className={`text-sm font-semibold mt-0.5 ${valueClass ?? cls(deluxe, "text-blue-800", "text-blue-300")}`}>
        {value != null ? Number(value).toFixed(4) : "—"}
      </p>
    </div>
  );
}

export function ConfusionMatrix({
  labels,
  matrix,
}: {
  labels: string[];
  matrix: number[][];
}) {
  const maxCell = Math.max(1, ...matrix.flat());
  return (
    <table className="text-xs border-collapse">
      <thead>
        <tr>
          <th className="px-2 py-1"></th>
          {labels.map((label) => (
            <th key={label} className="px-2 py-1 font-mono whitespace-nowrap">
              {label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {matrix.map((row, rowIdx) => (
          <tr key={labels[rowIdx] ?? rowIdx}>
            <th className="px-2 py-1 text-left font-mono whitespace-nowrap">{labels[rowIdx]}</th>
            {row.map((cell, colIdx) => {
              const intensity = Math.max(0.08, Number(cell) / maxCell);
              const isDiagonal = rowIdx === colIdx;
              return (
                <td
                  key={`${rowIdx}-${colIdx}`}
                  className="px-2 py-1 text-center font-semibold"
                  style={{
                    backgroundColor: isDiagonal
                      ? `rgba(34, 197, 94, ${intensity})`
                      : `rgba(59, 130, 246, ${intensity})`,
                  }}
                >
                  {cell}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
