import { useState, useEffect } from "react";
import { AppHeader } from "@/components/AppHeader";
import { Guide } from "@/components/Guide";
import { TrainingView } from "@/components/TrainingView";
import { useAppUi } from "@/hooks/useAppUi";
import { useSessionTransfer } from "@/hooks/useSessionTransfer";
import { useTrainingPipeline } from "@/hooks/useTrainingPipeline";
import { loadPersisted } from "@/hooks/trainingPipelineUtils";
import { sessionHeaders } from "@/lib/api";

export default function MediaFeatureLabPro() {
  const [showRestoredToast, setShowRestoredToast] = useState(false);
  const savedPipeline = loadPersisted();
  const { lang, setLang, deluxe, setDeluxe, showGuide, setShowGuide, closeGuide } = useAppUi();

  const i18n = {
    cs: {
      appTitle: "Media Feature Lab — Pro",
      appSubtitle: "Vysoká škola ekonomická v Praze",
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
      appTitle: "Media Feature Lab — Pro",
      appSubtitle: "Prague University of Economics and Business",
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
  const { importInputRef, handleExportSession, handleImportSession } = useSessionTransfer(
    t.transferError,
    t.importOk,
  );

  useEffect(() => {
    if (!pipeline.isRestoring && pipeline.restoredWithData) {
      setShowRestoredToast(true);
      const timer = setTimeout(() => setShowRestoredToast(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [pipeline.isRestoring, pipeline.restoredWithData]);

  async function handleReset() {
    if (!window.confirm(t.resetConfirm)) return;
    try {
      localStorage.removeItem("mflFilesMeta");
      localStorage.removeItem("mflFileType");
      localStorage.removeItem("mflGuideSeen");
    } catch {
      // localStorage unavailable
    }
    await pipeline.resetPipeline();
    window.location.reload();
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

        <AppHeader
          deluxe={deluxe}
          lang={lang}
          setLang={setLang}
          setDeluxe={setDeluxe}
          setShowGuide={setShowGuide}
          handleReset={handleReset}
          handleExportSession={handleExportSession}
          triggerImport={() => importInputRef.current?.click()}
          t={t}
        />

        {pipeline.isRestoring && (savedPipeline?.trainingStep ?? 1) > 1 && (
          <div className="flex items-center gap-2 text-sm text-slate-400 mb-4">
            <span className="inline-block w-3 h-3 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
            {lang === "cs" ? "Obnovuji relaci..." : "Restoring session..."}
          </div>
        )}

        <TrainingView
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
          isExtracting={pipeline.extractionBusy}
          trainingDataX={pipeline.trainingDataX}
          datasetYColumns={pipeline.datasetYColumns}
          onTrain={pipeline.handleTrain}
          isTraining={pipeline.trainingBusy}
          trainResult={pipeline.trainResult}
          onExtractTesting={pipeline.handleExtractTesting}
          isExtractingTest={pipeline.testExtractionBusy}
          testingDataX={pipeline.testingDataX}
          onPredict={pipeline.handlePredict}
          isPredicting={pipeline.predictBusy}
          predictions={pipeline.predictions}
          predictionMetrics={pipeline.predictionMetrics}
          modelProvider={pipeline.modelProvider}
          setModelProvider={pipeline.setModelProvider}
          llmEndpoint={pipeline.llmEndpoint}
          setLlmEndpoint={pipeline.setLlmEndpoint}
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

      {showRestoredToast && (
        <div className="fixed bottom-4 right-4 bg-slate-700 text-white text-sm px-4 py-2 rounded-lg shadow-lg z-50">
          {lang === "cs" ? "Relace obnovena" : "Session restored"}
        </div>
      )}
    </div>
  );
}
