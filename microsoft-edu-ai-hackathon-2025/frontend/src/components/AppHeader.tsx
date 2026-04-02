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
            Media Feature Lab — Pro
          </h1>
          <p className={`mt-0.5 text-sm ${deluxe ? "text-slate-300" : "text-slate-600"}`}>
            Prague University of Economics and Business
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
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

        <Button variant="outline" onClick={triggerImport}>
          {t.importSession}
        </Button>
      </div>
    </div>
  );
}
