import { useRef } from "react";
import { EXPORT_SESSION_URL, IMPORT_SESSION_URL, sessionHeaders } from "@/lib/api";

export function useSessionTransfer(transferErrorLabel: string, importOkLabel: string) {
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
      const msg = e instanceof Error ? e.message : String(e);
      alert(`${transferErrorLabel}: ${msg}`);
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
      alert(importOkLabel);
      window.location.reload();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      alert(`${transferErrorLabel}: ${msg}`);
    }
  }

  return {
    importInputRef,
    handleExportSession,
    handleImportSession,
  };
}
