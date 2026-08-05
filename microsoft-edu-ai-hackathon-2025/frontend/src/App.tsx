import { useState, useEffect } from "react";
import { ExternalLink, ShieldCheck } from "lucide-react";
import { AppHeader } from "@/components/AppHeader";
import { Guide } from "@/components/Guide";
import { TrainingView } from "@/components/TrainingView";
import { Button } from "@/components/ui/button";
import { useAppUi } from "@/hooks/useAppUi";
import { useSessionTransfer } from "@/hooks/useSessionTransfer";
import { useTrainingPipeline } from "@/hooks/useTrainingPipeline";
import { loadPersisted } from "@/hooks/trainingPipelineUtils";

function UsageAgreementDialog({
  deluxe,
  lang,
  onAccept,
}: {
  deluxe: boolean;
  lang: "cs" | "en";
  onAccept: () => void;
}) {
  const disclaimerHref = lang === "en" ? "/disclaimer-en.html" : "/disclaimer-cs.html";
  const text = lang === "en"
    ? {
        title: "Before You Start",
        subtitle: "Data protection and fair use",
        intro:
          "Before uploading media, confirm that the files do not contain personal, confidential, proprietary, or sensitive data.",
        point1: "Use only data you are allowed to process.",
        point2: "Anonymize files and labels before analysis.",
        point3: "Review generated outputs before using them in reports or decisions.",
        link: "Read full terms",
        accept: "I understand and want to continue",
      }
    : {
        title: "Než začnete",
        subtitle: "Ochrana dat a férové použití",
        intro:
          "Před nahráním médií potvrďte, že soubory neobsahují osobní, důvěrné, proprietární ani citlivé údaje.",
        point1: "Používejte jen data, která smíte zpracovávat.",
        point2: "Před analýzou anonymizujte soubory i labely.",
        point3: "Vygenerované výstupy před použitím v práci nebo rozhodování zkontrolujte.",
        link: "Přečíst celé podmínky",
        accept: "Rozumím a chci pokračovat",
      };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-slate-950/45 backdrop-blur-sm" />
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="usage-agreement-title"
        className={`relative z-10 w-full max-w-lg rounded-2xl border p-5 shadow-2xl ${
          deluxe
            ? "border-slate-700 bg-slate-900 text-white"
            : "border-slate-200 bg-white text-slate-950"
        }`}
      >
        <div className="flex items-start gap-3">
          <div className={`rounded-xl p-2 ${deluxe ? "bg-blue-500/15 text-blue-200" : "bg-blue-50 text-blue-700"}`}>
            <ShieldCheck className="h-5 w-5" aria-hidden="true" />
          </div>
          <div>
            <p className={`text-xs font-semibold uppercase ${deluxe ? "text-blue-200" : "text-blue-700"}`}>
              {text.subtitle}
            </p>
            <h2 id="usage-agreement-title" className="mt-1 text-xl font-semibold">
              {text.title}
            </h2>
          </div>
        </div>

        <p className={`mt-4 text-sm leading-relaxed ${deluxe ? "text-slate-300" : "text-slate-600"}`}>
          {text.intro}
        </p>

        <ul className={`mt-4 list-disc space-y-2 pl-5 text-sm ${deluxe ? "text-slate-200" : "text-slate-700"}`}>
          <li>{text.point1}</li>
          <li>{text.point2}</li>
          <li>{text.point3}</li>
        </ul>

        <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <a
            href={disclaimerHref}
            target="_blank"
            rel="noopener noreferrer"
            className={`inline-flex items-center gap-1.5 text-sm font-medium underline-offset-4 hover:underline ${
              deluxe ? "text-blue-200 hover:text-white" : "text-blue-700 hover:text-blue-900"
            }`}
          >
            {text.link}
            <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
          </a>
          <Button onClick={onAccept}>{text.accept}</Button>
        </div>
      </section>
    </div>
  );
}

export default function MediaFeatureLabPro() {
  const [showRestoredToast, setShowRestoredToast] = useState(false);
  const [usageAgreementAccepted, setUsageAgreementAccepted] = useState(() => {
    try {
      return localStorage.getItem("mflUsageAgreementAccepted") === "1";
    } catch {
      return false;
    }
  });
  const savedPipeline = loadPersisted();
  const { lang, setLang, deluxe, setDeluxe, showGuide, setShowGuide, closeGuide } = useAppUi();

  const i18n = {
    cs: {
      appTitle: "Media Feature Lab — Pro",
      appSubtitle: "Vysoká škola ekonomická v Praze",
      toggleTheme: "Přepnout motiv",
      guide: "Video průvodce",
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
      guide: "Video guide",
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
  const [transferNotice, setTransferNotice] = useState<
    { message: string; type: "error" | "success" } | null
  >(null);
  const { importInputRef, handleExportSession, handleImportSession } = useSessionTransfer(
    t.transferError,
    t.importOk,
    (message, type) => setTransferNotice({ message, type }),
  );

  useEffect(() => {
    if (!transferNotice) return;
    const ms = transferNotice.type === "error" ? 6000 : 3000;
    const timer = setTimeout(() => setTransferNotice(null), ms);
    return () => clearTimeout(timer);
  }, [transferNotice]);

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

  function handleAcceptUsageAgreement() {
    setUsageAgreementAccepted(true);
    try {
      localStorage.setItem("mflUsageAgreementAccepted", "1");
    } catch {
      // localStorage unavailable
    }
  }

  return (
    <div
      className={`min-h-screen ${
        deluxe
          ? "dark bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-white"
          : "bg-slate-50 text-slate-900"
      }`}
    >
      <div className="mx-auto max-w-6xl px-6 py-6">
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
          usageAgreementAccepted={usageAgreementAccepted}
        />
      </div>

      {showGuide && <Guide deluxe={deluxe} onClose={closeGuide} uiLanguage={lang} />}

      {!usageAgreementAccepted && (
        <UsageAgreementDialog
          deluxe={deluxe}
          lang={lang}
          onAccept={handleAcceptUsageAgreement}
        />
      )}

      {showRestoredToast && (
        <div className="fixed bottom-4 right-4 bg-slate-700 text-white text-sm px-4 py-2 rounded-lg shadow-lg z-50">
          {lang === "cs" ? "Relace obnovena" : "Session restored"}
        </div>
      )}

      {transferNotice && (
        <div
          role={transferNotice.type === "error" ? "alert" : "status"}
          aria-live={transferNotice.type === "error" ? "assertive" : "polite"}
          className={`fixed bottom-16 right-4 flex items-start gap-3 max-w-sm text-sm px-4 py-2 rounded-lg shadow-lg z-50 ${
            transferNotice.type === "error"
              ? "bg-red-600 text-white"
              : "bg-emerald-600 text-white"
          }`}
        >
          <span>{transferNotice.message}</span>
          <button
            type="button"
            onClick={() => setTransferNotice(null)}
            aria-label={lang === "cs" ? "Zavřít" : "Dismiss"}
            className="shrink-0 text-lg leading-none opacity-80 hover:opacity-100"
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
}
