export function enrichError(raw: string): { message: string; hint?: string } {
  if (raw.includes("No data remained after joining")) {
    return {
      message: raw,
      hint:
        'Zkontroluj, že názvy souborů v CSV (první sloupec bez přípony) odpovídají názvům médií v ZIPu. Např. soubor "video.mp4" → řádek CSV musí mít hodnotu "video".',
    };
  }
  if (raw.includes("Column '") && raw.includes("not found")) {
    return {
      message: raw,
      hint: "Dostupné sloupce jsou vypsány v chybě výše — zkopíruj přesný název.",
    };
  }
  if (raw.includes("ZIP contains no media files")) {
    return {
      message: raw,
      hint: "ZIP musí obsahovat videa (.mp4, .avi, .mov, .mkv) nebo obrázky (.jpg, .png, .webp, .gif).",
    };
  }
  if (raw.includes("Phase 2") && raw.includes("must be completed")) {
    return { message: raw, hint: "Vrať se na Fázi 2 a dokonči extrakci trénovacích dat." };
  }
  if (raw.includes("Model is not trained")) {
    return { message: raw, hint: "Nejprve dokonči Fázi 3 (Trénink)." };
  }
  if (raw.includes("Missing dataset_Y")) {
    return {
      message: raw,
      hint: "CSV s labels musí být součástí ZIPu nebo ho nahraj samostatně přes checkbox níže.",
    };
  }
  return { message: raw };
}
