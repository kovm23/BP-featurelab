import { useEffect, useRef } from "react";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export function Guide({
  onClose,
  uiLanguage = "cs",
  deluxe = false,
}: {
  onClose: () => void;
  uiLanguage?: "cs" | "en";
  deluxe?: boolean;
}) {
  const cardRef = useRef<HTMLDivElement>(null);

  const t = uiLanguage === "en"
    ? {
      title: "How to use Media Feature Lab",
      subtitle: "Quick walkthrough of the 5-phase pipeline",
      close: "Got it",
      pipelineCard: "The training pipeline has 5 phases. Finish each phase before moving to the next one. Progress is saved automatically, so you can continue after refresh or restart.",
      phase1Title: "Phase 1 — Feature Discovery",
      phase1Desc: "Upload sample media, set target variable and target type (regression/classification). AI proposes a feature specification.",
      phase2Title: "Phase 2 — Training Feature Extraction",
      phase2Desc: "Review/edit feature spec and run extraction on training media. The output is training dataset_X.",
      phase3Title: "Phase 3 — Model Training",
      phase3Desc: "Select target column from labels and train the model. You get metrics, feature importance and rules.",
      phase4Title: "Phase 4 — Testing Feature Extraction",
      phase4Desc: "Extract the same features from unseen testing media.",
      phase5Title: "Phase 5 — Prediction & Evaluation",
      phase5Desc: "Run predictions, inspect applied rule per row, and evaluate metrics when test labels are provided.",
      tipsTitle: "Practical tips",
      tip1: "Use Export session / Import session to move work to another server.",
      tip2: "Choose target mode at the beginning: continuous target = regression, categorical target = classification.",
      tip3: "Download outputs after each phase for traceability and reporting.",
    }
    : {
      title: "Jak používat Media Feature Lab",
      subtitle: "Rychlý průvodce 5fázovou pipeline",
      close: "Rozumím",
      pipelineCard: "Trénovací pipeline má 5 fází. Každou fázi je potřeba dokončit, než přejdete na další. Postup se ukládá automaticky, takže po obnovení stránky nebo restartu navážete tam, kde jste skončili.",
      phase1Title: "Fáze 1 — Objevování featur",
      phase1Desc: "Nahrajte ukázková média, nastavte cílovou proměnnou a typ cíle (regrese/klasifikace). AI navrhne specifikaci featur.",
      phase2Title: "Fáze 2 — Extrakce trénovacích featur",
      phase2Desc: "Zkontrolujte/upravte feature spec a spusťte extrakci na trénovacích médiích. Výstupem je trénovací dataset_X.",
      phase3Title: "Fáze 3 — Trénování modelu",
      phase3Desc: "Vyberte cílový sloupec z labels a natrénujte model. Získáte metriky, důležitost featur a pravidla.",
      phase4Title: "Fáze 4 — Extrakce testovacích featur",
      phase4Desc: "Extrahujte stejné featury z dosud neviděných testovacích médií.",
      phase5Title: "Fáze 5 — Predikce a vyhodnocení",
      phase5Desc: "Spusťte predikce, zkontrolujte použité pravidlo pro každý řádek a vyhodnoťte metriky při nahrání testovacích labels.",
      tipsTitle: "Praktické tipy",
      tip1: "Pro přesun práce na jiný server použijte Export relace / Import relace.",
      tip2: "Typ cíle vybírejte na začátku: spojitá hodnota = regrese, kategorická hodnota = klasifikace.",
      tip3: "Po každé fázi si můžete stáhnout výstupy pro audit a reporting.",
    };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    cardRef.current?.focus();
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/30 backdrop-blur-sm"
        onClick={onClose}
      />
      <Card
        ref={cardRef}
        tabIndex={-1}
        className={`relative z-10 max-w-2xl w-full max-h-[85vh] overflow-y-auto outline-none ${
          deluxe
            ? "border-slate-700 bg-slate-900"
            : "border-slate-200 bg-white"
        }`}
      >
        <CardHeader className="pb-2">
          <CardTitle className={deluxe ? "text-xl text-white" : "text-xl text-slate-900"}>
            {t.title}
          </CardTitle>
          <CardDescription className={deluxe ? "text-slate-300" : "text-slate-600"}>
            {t.subtitle}
          </CardDescription>
        </CardHeader>
        <CardContent className={`space-y-3 text-sm leading-relaxed ${deluxe ? "text-slate-100" : "text-slate-800"}`}>
          <div className={`rounded-lg p-3 border ${deluxe ? "bg-indigo-950/30 border-indigo-800" : "bg-indigo-50 border-indigo-200"}`}>
            {t.pipelineCard}
          </div>

          <div className={deluxe ? "rounded-lg p-3 bg-white/5" : "rounded-lg p-3 bg-slate-50"}>
            <b>{t.phase1Title}</b>
            <p className={`mt-1 ${deluxe ? "text-slate-300" : "text-slate-600"}`}>{t.phase1Desc}</p>
          </div>
          <div className={deluxe ? "rounded-lg p-3 bg-white/5" : "rounded-lg p-3 bg-slate-50"}>
            <b>{t.phase2Title}</b>
            <p className={`mt-1 ${deluxe ? "text-slate-300" : "text-slate-600"}`}>{t.phase2Desc}</p>
          </div>
          <div className={deluxe ? "rounded-lg p-3 bg-white/5" : "rounded-lg p-3 bg-slate-50"}>
            <b>{t.phase3Title}</b>
            <p className={`mt-1 ${deluxe ? "text-slate-300" : "text-slate-600"}`}>{t.phase3Desc}</p>
          </div>
          <div className={deluxe ? "rounded-lg p-3 bg-white/5" : "rounded-lg p-3 bg-slate-50"}>
            <b>{t.phase4Title}</b>
            <p className={`mt-1 ${deluxe ? "text-slate-300" : "text-slate-600"}`}>{t.phase4Desc}</p>
          </div>
          <div className={deluxe ? "rounded-lg p-3 bg-white/5" : "rounded-lg p-3 bg-slate-50"}>
            <b>{t.phase5Title}</b>
            <p className={`mt-1 ${deluxe ? "text-slate-300" : "text-slate-600"}`}>{t.phase5Desc}</p>
          </div>

          <div className={`rounded-lg p-3 border ${deluxe ? "bg-amber-900/20 border-amber-800" : "bg-amber-50 border-amber-200"}`}>
            <b>{t.tipsTitle}</b>
            <ul className="mt-1 space-y-1 list-disc pl-5">
              <li>{t.tip1}</li>
              <li>{t.tip2}</li>
              <li>{t.tip3}</li>
            </ul>
          </div>

          <div className="pt-2">
            <Button onClick={onClose} className="w-full">
              {t.close}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
