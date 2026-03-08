import React from "react";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export function Guide({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/30 backdrop-blur-sm"
        onClick={onClose}
      />
      <Card className="relative z-10 max-w-2xl w-full border-slate-200 bg-white dark:bg-slate-900 dark:border-slate-800">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl text-slate-900 dark:text-white">
            Jak používat Media Feature Lab
          </CardTitle>
          <CardDescription className="text-slate-600 dark:text-slate-300">
            Krátký průvodce pro první použití
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm leading-relaxed text-slate-800 dark:text-slate-100">
          <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
            <b>1) Připrav si soubory</b> — nahraj vždy <i>jeden typ</i> (PDF{" "}
            <b>nebo</b> obrázky <b>nebo</b> video <b>nebo</b> ZIP).
          </div>
          <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
            <b>2) Klasifikace (NOVÉ)</b> — Zadej kategorie (např. "Podvod,
            Pravda"), pokud chceš video automaticky roztřídit.
          </div>
          <div className="rounded-lg p-3 bg-slate-50 dark:bg-white/5">
            <b>3) Spusť extrakci</b> — Vyber si model (Qwen 2.5 je
            doporučený).
          </div>
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
