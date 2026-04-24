import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { fetchJson, REPEATABILITY_TEST_URL, sessionHeaders, STATUS_URL } from "@/lib/api";
import { cls } from "./shared";

interface FeatureStat {
  feature: string;
  type: "numeric" | "categorical";
  values: (string | number)[];
  mean?: number | null;
  std?: number;
  cv_pct?: number | null;
  mode?: string | null;
  mode_frequency?: number;
  mode_frequency_pct?: number;
}

interface RepeatabilityResult {
  n_repetitions: number;
  filename: string;
  model: string;
  feature_stats: FeatureStat[];
}

function CvBadge({ cv }: { cv: number | null | undefined }) {
  if (cv == null) return <span className="text-slate-400">—</span>;
  const color = cv < 10 ? "text-green-600" : cv < 20 ? "text-yellow-600" : "text-red-500";
  return <span className={`font-semibold ${color}`}>{cv.toFixed(1)}%</span>;
}

export function RepeatabilityTestModal({
  deluxe,
  modelProvider,
}: {
  deluxe: boolean;
  modelProvider?: string;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [nReps, setNReps] = useState(3);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("");
  const [result, setResult] = useState<RepeatabilityResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  function reset() {
    setFile(null);
    setProgress(0);
    setStage("");
    setResult(null);
    setError(null);
    setIsRunning(false);
  }

  function close() {
    abortRef.current?.abort();
    setIsOpen(false);
    reset();
  }

  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  async function handleRun() {
    if (!file) return;
    setIsRunning(true);
    setResult(null);
    setError(null);
    setProgress(0);
    setStage("Uploading...");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("n_repetitions", String(nReps));
    if (modelProvider) formData.append("model", modelProvider);

    try {
      const { job_id } = await fetchJson<{ job_id: string }>(REPEATABILITY_TEST_URL, {
        method: "POST",
        headers: sessionHeaders(),
        body: formData,
      });

      abortRef.current = new AbortController();
      const signal = abortRef.current.signal;

      while (!signal.aborted) {
        await new Promise((r) => setTimeout(r, 1500));
        if (signal.aborted) break;
        const status = await fetchJson<{
          progress: number;
          stage?: string;
          done?: boolean;
          error?: string;
          details?: RepeatabilityResult;
        }>(STATUS_URL(job_id), { signal, cache: "no-store", headers: sessionHeaders() });

        setProgress(status.progress ?? 0);
        setStage(status.stage ?? "");

        if (status.done || (status.progress ?? 0) >= 100) {
          if (status.error) {
            setError(status.error);
          } else if (status.details) {
            setResult(status.details as RepeatabilityResult);
          }
          break;
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "AbortError") {
        setError(err.message);
      }
    } finally {
      setIsRunning(false);
    }
  }

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className={`px-3 py-1.5 rounded text-xs border ${cls(
          deluxe,
          "border-slate-300 text-slate-600 hover:bg-slate-100",
          "border-slate-600 text-slate-300 hover:bg-slate-700"
        )}`}
      >
        LLM Repeatability Test
      </button>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div
        className={`rounded-xl shadow-2xl p-5 w-full max-w-2xl max-h-[85vh] overflow-y-auto ${cls(
          deluxe,
          "bg-white text-slate-900",
          "bg-slate-900 text-white border border-slate-700"
        )}`}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-bold text-base">LLM Repeatability Test</h2>
          <button onClick={close} className="p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-700">
            <X className="w-4 h-4" />
          </button>
        </div>

        <p className={`text-xs mb-3 ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>
          Runs feature extraction N times on one file and reports per-feature variance.
          Each run calls the LLM once (~1–3 min per run depending on model and video length).
        </p>

        <div className="space-y-3 mb-4">
          <div>
            <label className="text-xs font-medium block mb-1">Media file</label>
            <input
              type="file"
              accept="video/*,image/*,audio/*"
              disabled={isRunning}
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className={`text-xs w-full ${cls(deluxe, "text-slate-700", "text-slate-300")}`}
            />
          </div>
          <div>
            <label className="text-xs font-medium block mb-1">
              Repetitions: <strong>{nReps}</strong>
            </label>
            <input
              type="range"
              min={2}
              max={10}
              value={nReps}
              disabled={isRunning}
              onChange={(e) => setNReps(Number(e.target.value))}
              className="w-full"
            />
            <div className={`flex justify-between text-[10px] ${cls(deluxe, "text-slate-400", "text-slate-500")}`}>
              <span>2</span><span>10</span>
            </div>
          </div>
        </div>

        <button
          onClick={handleRun}
          disabled={!file || isRunning}
          className="w-full py-2 rounded text-sm font-semibold bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {isRunning ? "Running..." : "Start Test"}
        </button>

        {isRunning && (
          <div className="mt-4">
            <div className="h-1.5 rounded-full bg-slate-200 overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className={`text-xs mt-1 ${cls(deluxe, "text-slate-500", "text-slate-400")}`}>{stage}</p>
          </div>
        )}

        {error && (
          <p className="mt-3 text-xs text-red-500">Error: {error}</p>
        )}

        {result && (
          <div className="mt-4">
            <p className={`text-xs mb-2 font-medium ${cls(deluxe, "text-slate-600", "text-slate-400")}`}>
              Results — {result.n_repetitions} runs on {result.filename}
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className={cls(deluxe, "text-slate-500", "text-slate-400")}>
                    <th className="px-2 py-1 text-left">Feature</th>
                    <th className="px-2 py-1 text-left">Type</th>
                    <th className="px-2 py-1 text-left">Values</th>
                    <th className="px-2 py-1 text-left">Mean ± Std</th>
                    <th className="px-2 py-1 text-left">CV%</th>
                  </tr>
                </thead>
                <tbody>
                  {result.feature_stats.map((stat) => (
                    <tr
                      key={stat.feature}
                      className={`border-t ${cls(deluxe, "border-slate-200", "border-slate-700")}`}
                    >
                      <td className="px-2 py-1 font-mono font-medium">{stat.feature}</td>
                      <td className="px-2 py-1 text-slate-400">{stat.type}</td>
                      <td className="px-2 py-1 text-slate-400 max-w-[140px] truncate" title={stat.values.join(", ")}>
                        {stat.values.join(", ")}
                      </td>
                      <td className="px-2 py-1">
                        {stat.type === "numeric"
                          ? `${stat.mean?.toFixed(2) ?? "—"} ± ${stat.std?.toFixed(2) ?? "—"}`
                          : <span className="text-slate-500">{stat.mode} ({stat.mode_frequency_pct?.toFixed(0)}%)</span>
                        }
                      </td>
                      <td className="px-2 py-1">
                        {stat.type === "numeric"
                          ? <CvBadge cv={stat.cv_pct} />
                          : <span className="text-slate-400">—</span>
                        }
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className={`mt-2 text-[10px] ${cls(deluxe, "text-slate-400", "text-slate-500")}`}>
              CV% (coefficient of variation): green &lt;10%, yellow 10–20%, red &gt;20%
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
