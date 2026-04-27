import type { LlmEndpointConfig } from "@/lib/api";

export function LlmEndpointConfigPanel({
  deluxe,
  value,
  onChange,
}: {
  deluxe: boolean;
  value: LlmEndpointConfig;
  onChange: (cfg: LlmEndpointConfig) => void;
}) {
  const inputCls = `w-full p-2 text-sm rounded-lg border outline-none ${
    deluxe
      ? "bg-slate-900 border-slate-700 text-white focus:border-blue-400"
      : "bg-slate-50 border-slate-200 text-slate-900 focus:border-blue-400"
  }`;
  const labelCls = `block text-xs font-medium mb-1 ${deluxe ? "text-slate-400" : "text-slate-500"}`;
  const temperature = value.temperature ?? 0.1;

  return (
    <div className={`rounded-lg border p-3 space-y-3 ${deluxe ? "border-slate-700 bg-slate-800/40" : "border-slate-200 bg-slate-50"}`}>
      <p className={`text-xs ${deluxe ? "text-slate-400" : "text-slate-500"}`}>
        OpenAI-compatible endpoint (e.g. litellm.vse.cz). Replaces the local Ollama server.
      </p>

      <div>
        <label className={labelCls}>Base URL</label>
        <input
          type="url"
          placeholder="https://litellm.vse.cz/"
          value={value.baseUrl}
          onChange={(e) => onChange({ ...value, baseUrl: e.target.value })}
          className={inputCls}
        />
      </div>

      <div>
        <label className={labelCls}>API key</label>
        <input
          type="password"
          placeholder="sk-..."
          value={value.apiKey}
          onChange={(e) => onChange({ ...value, apiKey: e.target.value })}
          className={inputCls}
        />
      </div>

      <div>
        <label className={labelCls}>Model name</label>
        <input
          type="text"
          placeholder="e.g. qwen3.6-35b"
          value={value.model}
          onChange={(e) => onChange({ ...value, model: e.target.value })}
          className={inputCls}
        />
      </div>

      <div>
        <label className={labelCls}>
          Temperature: <span className="font-mono font-bold">{temperature.toFixed(2)}</span>
        </label>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={temperature}
          onChange={(e) => onChange({ ...value, temperature: parseFloat(e.target.value) })}
          className="w-full accent-blue-500"
        />
        <div className={`flex justify-between text-[10px] mt-0.5 ${deluxe ? "text-slate-500" : "text-slate-400"}`}>
          <span>0.0 (deterministic)</span>
          <span>1.0 (creative)</span>
        </div>
      </div>
    </div>
  );
}
