import { cls } from "./style";

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
              {cols.map((column) => (
                <th
                  key={column}
                  className={`px-2 py-1 text-left font-mono whitespace-nowrap ${cls(deluxe, "text-slate-600", "text-slate-400")}`}
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, index) => (
              <tr
                key={index}
                className={index % 2 === 0 ? cls(deluxe, "bg-white", "bg-slate-800/50") : cls(deluxe, "bg-slate-50/50", "bg-slate-900/50")}
              >
                {cols.map((column) => (
                  <td key={column} className={`px-2 py-1 whitespace-nowrap ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
                    {String(row[column] ?? "")}
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
