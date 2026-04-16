import React, { useState } from "react";
import { Check, Copy, FileVideo, FileText, Archive } from "lucide-react";
import { Image as ImageIcon } from "lucide-react";
import type { FileType, ProcessingResult } from "@/lib/api";
import { TYPE_STYLES, EXT_GROUPS } from "@/lib/api";

// =====================================================
// FILE DETECTION & UTILITIES
// =====================================================

/** Normalise a caught value (which is `unknown` in TS) into a human-readable string. */
export function getErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

export function detectType(files: File[]): FileType | null {
  const getGroup = (name: string): FileType | null => {
    const lower = name.toLowerCase();
    if (EXT_GROUPS.image.some((e) => lower.endsWith(e))) return "image";
    if (EXT_GROUPS.video.some((e) => lower.endsWith(e))) return "video";
    if (EXT_GROUPS.text.some((e) => lower.endsWith(e))) return "text";
    if (EXT_GROUPS.archive.some((e) => lower.endsWith(e))) return "archive";
    return null;
  };
  const types = new Set<FileType>();
  for (const f of files) {
    const t = getGroup(f.name);
    if (!t) return null;
    types.add(t);
  }
  if (types.size !== 1) return null;
  return [...types][0];
}

export function b64ToBlob(b64: string, contentType: string) {
  const base64 = b64.includes(",") ? b64.split(",")[1] : b64;
  const byteChars = atob(base64);
  const byteNumbers = new Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++)
    byteNumbers[i] = byteChars.charCodeAt(i);
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: contentType });
}

export function downloadText(
  name: string,
  text: string,
  mime = "text/plain;charset=utf-8"
) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadXLSX(x: unknown, filename = "output.xlsx") {
  if (typeof x === "string") {
    try {
      const blob = b64ToBlob(
        x,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
      );
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      return;
    } catch {
      /* fallback below */
    }
  }
  downloadText(filename.replace(/\.xlsx$/, ".txt"), String(x));
}

export function humanSize(bytes: number) {
  if (!bytes && bytes !== 0) return "";
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return (
    (bytes / Math.pow(1024, i)).toFixed(1) + " " + ["B", "KB", "MB", "GB"][i]
  );
}

export function fileIcon(name: string) {
  const lower = name.toLowerCase();
  const cls = "h-7 w-7";
  if ([".mp4", ".mov", ".avi", ".mkv"].some((e) => lower.endsWith(e)))
    return <FileVideo className={cls} />;
  if ([".png", ".jpg", ".jpeg", ".webp"].some((e) => lower.endsWith(e)))
    return <ImageIcon className={cls} />;
  if (lower.endsWith(".zip")) return <Archive className={cls} />;
  return <FileText className={cls} />;
}

export function uploaderBorder(type: FileType | null) {
  if (!type) return "";
  return TYPE_STYLES[type].border;
}

export type FileKey = string;
export const fileKey = (f: File): FileKey =>
  `${f.name}__${f.size}__${f.lastModified}`;

export function mergeFiles(existing: File[], incoming: File[]): File[] {
  const map = new Map<FileKey, File>();
  for (const f of existing) map.set(fileKey(f), f);
  for (const f of incoming) map.set(fileKey(f), f);
  return Array.from(map.values());
}

export function getOutputs(
  proc: ProcessingResult | undefined
): Record<string, unknown> {
  return proc?.outputs || proc?.key_frame_analysis?.outputs || {};
}

export function getProcType(
  proc: ProcessingResult | undefined
): FileType | undefined {
  return proc?.type;
}

export function getTranscript(
  proc: ProcessingResult | undefined
): string | null {
  if (!proc) return null;
  if (proc.tabular_output) {
    const transcripts = Object.values(proc.tabular_output)
      .map((v) => v.transcript)
      .filter(Boolean)
      .join("\n\n--- NEXT FILE ---\n\n");
    if (transcripts) return transcripts;
  }
  if (proc.transcriptions && typeof proc.transcriptions === "object") {
    return Object.values(proc.transcriptions).join(
      "\n\n--- NEXT FILE ---\n\n"
    );
  }
  return (
    (typeof proc.transcription === "string" && proc.transcription) ||
    (typeof proc.transcript === "string" && proc.transcript) ||
    (typeof proc.asr?.text === "string" && proc.asr.text) ||
    (typeof proc.key_frame_analysis?.transcription === "string" &&
      proc.key_frame_analysis.transcription) ||
    null
  );
}

export function tileBg(type: FileType | null) {
  switch (type) {
    case "image":
      return { bg: "#EFF6FF", fg: "#1E3A8A" };
    case "video":
      return { bg: "#ECFDF5", fg: "#065F46" };
    case "text":
      return { bg: "#FEF2F2", fg: "#7F1D1D" };
    case "archive":
      return { bg: "#FFFBEB", fg: "#92400E" };
    default:
      return { bg: "#F1F5F9", fg: "#0F172A" };
  }
}

// =====================================================
// COPY BUTTON COMPONENT
// =====================================================
export function CopyButton({ getText, uiLanguage = "cs" }: { getText: () => string; uiLanguage?: "cs" | "en" }) {
  const [copied, setCopied] = useState(false);
  const labelCopy = uiLanguage === "en" ? "Copy" : "Kopírovat";
  const labelCopied = uiLanguage === "en" ? "Copied!" : "Zkopírováno";
  const titleLabel = uiLanguage === "en" ? "Copy to clipboard" : "Zkopírovat do schránky";
  return (
    <button
      className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs hover:bg-slate-100 dark:hover:bg-white/10"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(getText());
          setCopied(true);
          setTimeout(() => setCopied(false), 1200);
        } catch {
          /* clipboard unavailable */
        }
      }}
      title={titleLabel}
      aria-label={titleLabel}
    >
      {copied ? (
        <Check className="h-3.5 w-3.5" aria-hidden="true" />
      ) : (
        <Copy className="h-3.5 w-3.5" aria-hidden="true" />
      )}
      {copied ? labelCopied : labelCopy}
    </button>
  );
}
