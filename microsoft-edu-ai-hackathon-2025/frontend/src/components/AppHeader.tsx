import { HelpCircle, Moon, RefreshCw, Sun } from "lucide-react";
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
  return (
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
          <h1 className={`text-[28px] font-semibold tracking-tight ${deluxe ? "text-white" : "text-slate-900"}`}>
            {t.appTitle}
          </h1>
          <p className={`mt-0.5 text-sm ${deluxe ? "text-slate-300" : "text-slate-600"}`}>
            {t.appSubtitle}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className={`p-1 rounded-lg flex ${deluxe ? "bg-white/10" : "bg-slate-200"}`} role="group" aria-label={t.language}>
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
          size="icon"
          className="rounded-full"
          onClick={() => setDeluxe((v) => !v)}
          title={t.toggleTheme}
          aria-label={t.toggleTheme}
        >
          {deluxe ? <Sun className="h-4 w-4" aria-hidden="true" /> : <Moon className="h-4 w-4" aria-hidden="true" />}
        </Button>

        <Button
          variant={deluxe ? "secondary" : "default"}
          size="icon"
          className="rounded-full"
          onClick={() => setShowGuide(true)}
          title={t.guide}
          aria-label={t.guide}
        >
          <HelpCircle className="h-4 w-4" aria-hidden="true" />
        </Button>

        <Button variant="outline" onClick={handleReset}>
          <RefreshCw className="mr-2 h-4 w-4" /> {t.reset}
        </Button>

        <Button variant="outline" onClick={handleExportSession}>
          {t.exportSession}
        </Button>

        <Button variant="outline" onClick={triggerImport}>
          {t.importSession}
        </Button>

      </div>
    </div>
  );
}
