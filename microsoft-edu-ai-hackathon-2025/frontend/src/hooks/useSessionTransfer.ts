import { useRef } from "react";
import { EXPORT_SESSION_URL, IMPORT_SESSION_URL, sessionHeaders } from "@/lib/api";
import { getErrorMessage } from "@/lib/helpers";

export type TransferNotify = (message: string, type: "error" | "success") => void;

export function useSessionTransfer(
  transferErrorLabel: string,
  importOkLabel: string,
  onNotify: TransferNotify,
) {
  const importInputRef = useRef<HTMLInputElement | null>(null);

  async function handleExportSession() {
    try {
      const res = await fetch(EXPORT_SESSION_URL, {
        method: "GET",
        headers: sessionHeaders(),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `${res.status} ${res.statusText}`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "mfl_session_export.zip";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      onNotify(`${transferErrorLabel}: ${getErrorMessage(e)}`, "error");
    }
  }

  async function handleImportSession(file: File) {
    const form = new FormData();
    form.append("file", file, file.name);
    try {
      const res = await fetch(IMPORT_SESSION_URL, {
        method: "POST",
        headers: sessionHeaders(),
        body: form,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `${res.status} ${res.statusText}`);
      }
      onNotify(importOkLabel, "success");
      // Give the user a moment to read the confirmation before reloading.
      setTimeout(() => window.location.reload(), 1500);
    } catch (e) {
      onNotify(`${transferErrorLabel}: ${getErrorMessage(e)}`, "error");
    }
  }

  return {
    importInputRef,
    handleExportSession,
    handleImportSession,
  };
}
