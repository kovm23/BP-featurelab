import { useState } from "react";
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
  const [open, setOpen] = useState(!!(value.baseUrl || value.apiKey));
  const isActive = !!(value.baseUrl && value.apiKey);

  const inputCls = `w-full p-2 text-sm rounded-lg border outline-none ${
    deluxe
      ? "bg-slate-900 border-slate-700 text-white focus:border-blue-400"
      : "bg-slate-50 border-slate-200 text-slate-900 focus:border-blue-400"
  }`;
  const labelCls = `block text-xs font-medium mb-1 ${deluxe ? "text-slate-400" : "text-slate-500"}`;

  return (
    <div className={`rounded-lg border ${deluxe ? "border-slate-700" : "border-slate-200"}`}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={`w-full flex items-center justify-between px-3 py-2 text-sm rounded-lg ${
          deluxe ? "hover:bg-slate-800 text-slate-300" : "hover:bg-slate-50 text-slate-600"
        }`}
      >
        <span className="flex items-center gap-2">
          <span>Custom LLM endpoint</span>
          {isActive && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 font-medium">
              active
            </span>
          )}
        </span>
        <span className={`text-xs ${deluxe ? "text-slate-500" : "text-slate-400"}`}>
          {open ? "▲" : "▼"}
        </span>
      </button>

      {open && (
        <div className={`px-3 pb-3 space-y-3 border-t ${deluxe ? "border-slate-700" : "border-slate-200"}`}>
          <p className={`text-xs mt-2 ${deluxe ? "text-slate-400" : "text-slate-500"}`}>
            Configure an alternative OpenAI-compatible endpoint (e.g. litellm.vse.cz).
            Leave empty to use the server default (Ollama).
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
              placeholder="e.g. qwen3.6-35b (leave empty to use model selector above)"
              value={value.model}
              onChange={(e) => onChange({ ...value, model: e.target.value })}
              className={inputCls}
            />
          </div>

          {isActive && (
            <button
              type="button"
              onClick={() => onChange({ baseUrl: "", apiKey: "", model: "" })}
              className="text-xs text-red-500 hover:text-red-700"
            >
              Clear custom endpoint
            </button>
          )}
        </div>
      )}
    </div>
  );
}
