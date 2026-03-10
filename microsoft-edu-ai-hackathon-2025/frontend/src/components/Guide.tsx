import { useEffect, useRef, useState } from "react";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";

type Tab = "predict" | "train";

export function Guide({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [tab, setTab] = useState<Tab>("train");
  const cardRef = useRef<HTMLDivElement>(null);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  // Focus trap
  useEffect(() => {
    if (open) cardRef.current?.focus();
  }, [open]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/30 backdrop-blur-sm"
        onClick={onClose}
      />
      <Card ref={cardRef} tabIndex={-1} className="relative z-10 max-w-2xl w-full max-h-[85vh] overflow-y-auto border-slate-200 bg-white dark:bg-slate-900 dark:border-slate-800 outline-none">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl text-slate-900 dark:text-white">
            Jak používat Media Feature Lab
          </CardTitle>
          <CardDescription className="text-slate-600 dark:text-slate-300">
            Krátký průvodce pro první použití
          </CardDescription>
          <div className="flex gap-2 pt-2">
            <button
              onClick={() => setTab("train")}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                tab === "train"
                  ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200"
                  : "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400"
              }`}
            >
              Trénovací pipeline
            </button>
            <button
              onClick={() => setTab("predict")}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                tab === "predict"
                  ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200"
                  : "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400"
              }`}
            >
              Predikce
            </button>
          </div>
        </CardHeader>
        <CardContent className="space-y-3 text-sm leading-relaxed text-slate-800 dark:text-slate-100">
          {tab === "train" ? <TrainGuide /> : <PredictGuide />}
          <div className="pt-2">
            <Button onClick={onClose} className="w-full">
              Rozumím
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function TrainGuide() {
  return (
    <>
      <div className="rounded-lg p-3 bg-indigo-50 dark:bg-indigo-950/30 border border-indigo-200 dark:border-indigo-800">
        <b>Trénovací pipeline</b> se skládá z 5 fází. Každá fáze musí být
        dokončena, než můžeš pokračovat na další. Postup se automaticky ukládá
        — pokud zavřeš prohlížeč, po návratu budeš pokračovat tam, kde jsi
        skončil/a. Tlačítkem <i>Reset</i> se vrátíš na začátek.
      </div>

      <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
        <b>Fáze 1 — Objevování featur</b>
        <p className="mt-1 text-slate-600 dark:text-slate-300">
          Nahraj trénovací média a zadej cílovou proměnnou (např. „movie
          memorability score"). LLM analyzuje obsah a navrhne sadu featur,
          které mohou být relevantní pro predikci. Volitelně můžeš přidat CSV
          se štítky (dataset_Y) pro přesnější návrh.
        </p>
      </div>

      <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
        <b>Fáze 2 — Extrakce trénovacích featur</b>
        <p className="mt-1 text-slate-600 dark:text-slate-300">
          LLM extrahuje hodnoty navržených featur z každého trénovacího
          souboru. Výsledkem je tabulka (dataset_X), kde každý řádek odpovídá
          jednomu souboru. Opět lze přidat dataset_Y.
        </p>
      </div>

      <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
        <b>Fáze 3 — Trénování modelu (RuleKit)</b>
        <p className="mt-1 text-slate-600 dark:text-slate-300">
          Na extrahovaných datech se natrénuje RuleKit regresní model. Výsledkem
          jsou interpretovatelná pravidla (IF … THEN …) a metrika MSE na
          trénovacích datech.
        </p>
      </div>

      <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
        <b>Fáze 4 — Extrakce testovacích featur</b>
        <p className="mt-1 text-slate-600 dark:text-slate-300">
          Nahraj nová (testovací) média. LLM z nich extrahuje stejné featury
          jako v fázi 2. Tato data se použijí pro predikci.
        </p>
      </div>

      <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
        <b>Fáze 5 — Predikce a vyhodnocení</b>
        <p className="mt-1 text-slate-600 dark:text-slate-300">
          Natrénovaná pravidla se aplikují na testovací data. Pro každý soubor
          dostaneš predikovanou hodnotu a pravidlo, které se použilo. Pokud
          nahraješ i testovací dataset_Y, zobrazí se metriky přesnosti (MSE,
          MAE, korelace).
        </p>
      </div>
    </>
  );
}

function PredictGuide() {
  return (
    <>
      <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
        <b>1) Připrav si soubory</b> — nahraj vždy <i>jeden typ</i> (PDF{" "}
        <b>nebo</b> obrázky <b>nebo</b> video <b>nebo</b> ZIP).
      </div>
      <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
        <b>2) Klasifikace</b> — Zadej kategorie (např. „Podvod, Pravda"), pokud
        chceš video automaticky roztřídit.
      </div>
      <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
        <b>3) Spusť extrakci</b> — Vyber si model (Qwen 2.5 je doporučený).
      </div>
    </>
  );
}
