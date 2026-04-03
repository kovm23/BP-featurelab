export function enrichError(raw: string, uiLanguage: "cs" | "en" = "cs"): { message: string; hint?: string } {
  const t = uiLanguage === "en"
    ? {
        joinHint:
          'Check that file names in the CSV (first column without extension) match media names in the ZIP. Example: file "video.mp4" -> CSV value "video".',
        columnHint: "Available columns are listed in the error above. Copy the exact column name.",
        zipHint: "The ZIP must contain videos (.mp4, .avi, .mov, .mkv) or images (.jpg, .png, .webp, .gif).",
        phase2Hint: "Go back to Phase 2 and finish training data extraction.",
        trainingHint: "Finish Phase 3 (Training) first.",
        missingLabelsHint: "The labels CSV must be inside the ZIP or uploaded separately using the checkbox below.",
      }
    : {
        joinHint:
          'Zkontrolujte, že názvy souborů v CSV (první sloupec bez přípony) odpovídají názvům médií v ZIPu. Např. soubor "video.mp4" -> řádek CSV musí mít hodnotu "video".',
        columnHint: "Dostupné sloupce jsou vypsány v chybě výše. Zkopírujte přesný název.",
        zipHint: "ZIP musí obsahovat videa (.mp4, .avi, .mov, .mkv) nebo obrázky (.jpg, .png, .webp, .gif).",
        phase2Hint: "Vraťte se na Fázi 2 a dokončete extrakci trénovacích dat.",
        trainingHint: "Nejprve dokončete Fázi 3 (Trénink).",
        missingLabelsHint: "CSV s labely musí být součástí ZIPu nebo ho nahrajte samostatně přes checkbox níže.",
      };

  if (raw.includes("No data remained after joining")) {
    return {
      message: raw,
      hint: t.joinHint,
    };
  }
  if (raw.includes("Column '") && raw.includes("not found")) {
    return {
      message: raw,
      hint: t.columnHint,
    };
  }
  if (raw.includes("ZIP contains no media files")) {
    return {
      message: raw,
      hint: t.zipHint,
    };
  }
  if (raw.includes("Phase 2") && raw.includes("must be completed")) {
    return { message: raw, hint: t.phase2Hint };
  }
  if (raw.includes("Model is not trained")) {
    return { message: raw, hint: t.trainingHint };
  }
  if (raw.includes("Missing dataset_Y")) {
    return {
      message: raw,
      hint: t.missingLabelsHint,
    };
  }
  return { message: raw };
}
