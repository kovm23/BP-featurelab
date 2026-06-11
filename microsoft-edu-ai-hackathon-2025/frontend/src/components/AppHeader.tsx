import { CirclePlay, Download, Moon, RefreshCw, Sun, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";

export function AppHeader({
  deluxe,
  lang,
  setLang,
  setDeluxe,
  setShowGuide,
  handleReset,
  handleExportSession,
  triggerImport,
  t,
}: {
  deluxe: boolean;
  lang: "cs" | "en";
  setLang: (lang: "cs" | "en") => void;
  setDeluxe: (updater: (prev: boolean) => boolean) => void;
  setShowGuide: (value: boolean) => void;
  handleReset: () => void;
  handleExportSession: () => void;
  triggerImport: () => void;
  t: {
    toggleTheme: string;
    guide: string;
    reset: string;
    exportSession: string;
    importSession: string;
    language: string;
    appTitle: string;
    appSubtitle: string;
    schoolLogoAlt: string;
  };
}) {
  const headerOutlineButtonClass = deluxe
    ? "border-white/15 bg-white/10 text-white hover:bg-white/15 hover:text-white"
    : "";

  return (
    <header
      className={`mb-6 rounded-2xl border px-4 py-3 shadow-sm backdrop-blur ${
        deluxe
          ? "border-white/10 bg-white/10"
          : "border-slate-200/80 bg-white/85"
      }`}
    >
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex min-w-0 items-center gap-3">
          <a href="https://www.vse.cz/" target="_blank" rel="noopener noreferrer">
            <img
              src="/VSE_logo_CZ_circle_blue.png"
              alt={t.schoolLogoAlt}
              className="h-12 w-12 rounded-full bg-white shadow-sm ring-1 ring-slate-200"
            />
          </a>
          <div className="min-w-0">
            <h1 className={`truncate text-2xl font-semibold ${deluxe ? "text-white" : "text-slate-950"}`}>
              {t.appTitle}
            </h1>
            <p className={`mt-0.5 text-sm ${deluxe ? "text-slate-300" : "text-slate-600"}`}>
              {t.appSubtitle}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className={`flex rounded-lg p-1 ${deluxe ? "bg-white/10" : "bg-slate-100"}`} role="group" aria-label={t.language}>
            <button
              onClick={() => setLang("cs")}
              aria-pressed={lang === "cs"}
              aria-label="Přepnout jazyk na češtinu"
              className={`px-2 py-1 text-xs font-medium rounded-md transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
                lang === "cs"
                  ? deluxe
                    ? "bg-slate-700 text-white shadow"
                    : "bg-white text-slate-900 shadow"
                  : deluxe
                    ? "text-slate-400 hover:text-white"
                    : "text-slate-600 hover:text-slate-900"
              }`}
            >
              CZ
            </button>
            <button
              onClick={() => setLang("en")}
              aria-pressed={lang === "en"}
              aria-label="Switch language to English"
              className={`px-2 py-1 text-xs font-medium rounded-md transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
                lang === "en"
                  ? deluxe
                    ? "bg-slate-700 text-white shadow"
                    : "bg-white text-slate-900 shadow"
                  : deluxe
                    ? "text-slate-400 hover:text-white"
                    : "text-slate-600 hover:text-slate-900"
              }`}
            >
              EN
            </button>
          </div>

          <Button
            variant={deluxe ? "secondary" : "outline"}
            size="sm"
            className="h-9 w-9 rounded-lg p-0"
            onClick={() => setDeluxe((v) => !v)}
            title={t.toggleTheme}
            aria-label={t.toggleTheme}
          >
            {deluxe ? <Sun className="h-4 w-4" aria-hidden="true" /> : <Moon className="h-4 w-4" aria-hidden="true" />}
          </Button>

          <Button
            variant={deluxe ? "secondary" : "default"}
            size="sm"
            onClick={() => setShowGuide(true)}
            title={t.guide}
            aria-label={t.guide}
          >
            <CirclePlay className="h-4 w-4" aria-hidden="true" /> {t.guide}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={handleReset}
            className={headerOutlineButtonClass}
          >
            <RefreshCw className="mr-2 h-4 w-4" /> {t.reset}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={handleExportSession}
            className={headerOutlineButtonClass}
          >
            <Download className="mr-2 h-4 w-4" />
            {t.exportSession}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={triggerImport}
            className={headerOutlineButtonClass}
          >
            <Upload className="mr-2 h-4 w-4" />
            {t.importSession}
          </Button>
        </div>
      </div>
    </header>
  );
}
