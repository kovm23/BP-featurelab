import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { RefreshCw, HelpCircle, Moon, Sun } from "lucide-react";

import { Guide } from "@/components/Guide";
import { TrainingView } from "@/components/TrainingView";
import { useTrainingPipeline } from "@/hooks/useTrainingPipeline";
import { EXPORT_SESSION_URL, IMPORT_SESSION_URL, sessionHeaders } from "@/lib/api";

export default function MediaFeatureLabPro() {
  const [resetKey, setResetKey] = useState(0);
  const [lang, setLang] = useState<"cs" | "en">(() => {
    try {
      const saved = localStorage.getItem("mflLang");
      if (saved === "en" || saved === "cs") return saved;
      // Auto-detect from browser on first visit
      const browserLang = navigator.language || navigator.languages?.[0] || "cs";
      return browserLang.toLowerCase().startsWith("en") ? "en" : "cs";
    } catch {
      return "cs";
    }
  });

  const i18n = {
    cs: {
      toggleTheme: "Přepnout motiv",
      guide: "Průvodce",
      reset: "Reset",
      exportSession: "Export relace",
      importSession: "Import relace",
      language: "Jazyk",
      schoolLogoAlt: "Logo školy",
      importOk: "Relace byla naimportována. Aplikace se obnoví.",
      transferError: "Přenos relace selhal",
      resetConfirm: "Opravdu chcete resetovat celý pipeline? Všechna data budou ztracena.",
    },
    en: {
      toggleTheme: "Toggle theme",
      guide: "Help",
      reset: "Reset",
      exportSession: "Export session",
      importSession: "Import session",
      language: "Language",
      schoolLogoAlt: "School logo",
      importOk: "Session import completed. The app will reload.",
      transferError: "Session transfer failed",
      resetConfirm: "Do you really want to reset the whole pipeline? All data will be lost.",
    },
  } as const;
  const t = i18n[lang];

  const pipeline = useTrainingPipeline(lang);

  const [deluxe, setDeluxe] = useState<boolean>(() => {
    const saved = localStorage.getItem("mflTheme");
    if (saved === "dark") return true;
    if (saved === "light") return false;
    return false;
  });

  useEffect(() => {
    localStorage.setItem("mflTheme", deluxe ? "dark" : "light");
  }, [deluxe]);

  useEffect(() => {
    localStorage.setItem("mflLang", lang);
  }, [lang]);

  const [showGuide, setShowGuide] = useState(() => {
    try {
      return !localStorage.getItem("mflGuideSeen");
    } catch {
      return false;
    }
  });

  const closeGuide = useCallback(() => {
    setShowGuide(false);
    try {
      localStorage.setItem("mflGuideSeen", "1");
    } catch {
      // localStorage unavailable
    }
  }, []);

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
      alert(`${t.transferError}: ${msg}`);
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
      alert(t.importOk);
      window.location.reload();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      alert(`${t.transferError}: ${msg}`);
    }
  }

  function handleReset() {
    if (!window.confirm(t.resetConfirm)) return;
    pipeline.resetPipeline();
    try {
      localStorage.removeItem("mflFilesMeta");
      localStorage.removeItem("mflFileType");
      localStorage.removeItem("mflGuideSeen");
    } catch {
      // localStorage unavailable
    }
    setResetKey((k) => k + 1);
  }

  return (
    <div
      className={`min-h-screen ${
        deluxe
          ? "bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-white"
          : "bg-slate-50 text-slate-900"
      }`}
    >
      <div className="mx-auto max-w-5xl px-6 py-6">
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <a href="https://www.vse.cz/" target="_blank" rel="noopener noreferrer">
              <img
                src="/VSE_logo_CZ_circle_blue.png"
                alt={t.schoolLogoAlt}
                className="h-12 w-12 rounded-full shadow"
              />
            </a>
            <div>
              <h1
                className={`text-[28px] font-semibold tracking-tight ${
                  deluxe ? "text-white" : "text-slate-900"
                }`}
              >
                Media Feature Lab — Pro
              </h1>
              <p className={`mt-0.5 text-sm ${deluxe ? "text-slate-300" : "text-slate-600"}`}>
                Prague University of Economics and Business
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <input
              ref={importInputRef}
              type="file"
              accept=".zip"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  handleImportSession(file);
                }
                e.currentTarget.value = "";
              }}
            />

            <div className={`p-1 rounded-lg flex ${deluxe ? "bg-white/10" : "bg-slate-200"}`}>
              <button
                onClick={() => setLang("cs")}
                className={`px-2 py-1 text-xs font-medium rounded-md transition-all ${
                  lang === "cs"
                    ? deluxe
                      ? "bg-slate-700 text-white shadow"
                      : "bg-white text-slate-900 shadow"
                    : deluxe
                    ? "text-slate-400 hover:text-white"
                    : "text-slate-600 hover:text-slate-900"
                }`}
                title={t.language}
              >
                CZ
              </button>
              <button
                onClick={() => setLang("en")}
                className={`px-2 py-1 text-xs font-medium rounded-md transition-all ${
                  lang === "en"
                    ? deluxe
                      ? "bg-slate-700 text-white shadow"
                      : "bg-white text-slate-900 shadow"
                    : deluxe
                    ? "text-slate-400 hover:text-white"
                    : "text-slate-600 hover:text-slate-900"
                }`}
                title={t.language}
              >
                EN
              </button>
            </div>

            <Button
              variant={deluxe ? "secondary" : "outline"}
              size="icon"
              className="rounded-full"
              onClick={() => setDeluxe((v) => !v)}
              title={t.toggleTheme}
            >
              {deluxe ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>

            <Button
              variant={deluxe ? "secondary" : "default"}
              size="icon"
              className="rounded-full"
              onClick={() => setShowGuide(true)}
              title={t.guide}
            >
              <HelpCircle className="h-4 w-4" />
            </Button>

            <Button variant="outline" onClick={handleReset}>
              <RefreshCw className="mr-2 h-4 w-4" /> {t.reset}
            </Button>

            <Button variant="outline" onClick={handleExportSession}>
              {t.exportSession}
            </Button>

            <Button variant="outline" onClick={() => importInputRef.current?.click()}>
              {t.importSession}
            </Button>
          </div>
        </div>

        <TrainingView
          key={resetKey}
          deluxe={deluxe}
          onCancel={pipeline.handleCancelActive}
          onDiscoverStart={pipeline.handleDiscover}
          isDiscovering={pipeline.isDiscovering}
          targetVariable={pipeline.targetVariable}
          setTargetVariable={pipeline.setTargetVariable}
          targetMode={pipeline.targetMode}
          setTargetMode={pipeline.setTargetMode}
          featureSpec={pipeline.featureSpec}
          setFeatureSpec={pipeline.setFeatureSpec}
          onExtractTraining={pipeline.handleExtractTraining}
          onExtractTrainingLocal={pipeline.handleExtractTrainingLocal}
          isExtracting={pipeline.extractionBusy}
          trainingDataX={pipeline.trainingDataX}
          datasetYColumns={pipeline.datasetYColumns}
          onTrain={pipeline.handleTrain}
          isTraining={pipeline.trainingBusy}
          trainResult={pipeline.trainResult}
          onExtractTesting={pipeline.handleExtractTesting}
          onExtractTestingLocal={pipeline.handleExtractTestingLocal}
          isExtractingTest={pipeline.testExtractionBusy}
          testingDataX={pipeline.testingDataX}
          onPredict={pipeline.handlePredict}
          isPredicting={pipeline.predictBusy}
          predictions={pipeline.predictions}
          predictionMetrics={pipeline.predictionMetrics}
          modelProvider={pipeline.modelProvider}
          setModelProvider={pipeline.setModelProvider}
          step={pipeline.trainingStep}
          onGoToStep={(s) => pipeline.setTrainingStep(s as 1 | 2 | 3 | 4 | 5)}
          progress={pipeline.progress}
          progressLabel={pipeline.progressLabel}
          error={pipeline.error}
          clearError={pipeline.clearError}
          ollamaOk={pipeline.ollamaOk}
          recheckOllama={pipeline.recheckOllama}
          queueBusy={pipeline.queueBusy}
          queuedCount={pipeline.queuedCount}
          uiLanguage={lang}
        />
      </div>

      {showGuide && <Guide deluxe={deluxe} onClose={closeGuide} uiLanguage={lang} />}
    </div>
  );
}
