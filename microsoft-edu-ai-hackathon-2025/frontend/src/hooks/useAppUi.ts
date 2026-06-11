import { useCallback, useEffect, useState } from "react";

function getStoredTheme(): "dark" | "light" | null {
  try {
    const saved = localStorage.getItem("mflTheme");
    return saved === "dark" || saved === "light" ? saved : null;
  } catch {
    return null;
  }
}

function browserPrefersDark(): boolean {
  try {
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  } catch {
    return false;
  }
}

export function useAppUi() {
  const [lang, setLang] = useState<"cs" | "en">(() => {
    try {
      const saved = localStorage.getItem("mflLang");
      if (saved === "en" || saved === "cs") return saved;
      const browserLang = navigator.language || navigator.languages?.[0] || "cs";
      return browserLang.toLowerCase().startsWith("en") ? "en" : "cs";
    } catch {
      return "cs";
    }
  });

  const [hasStoredTheme, setHasStoredTheme] = useState(() => getStoredTheme() !== null);
  const [deluxe, setDeluxeState] = useState<boolean>(() => {
    const saved = getStoredTheme();
    if (saved === "dark") return true;
    if (saved === "light") return false;
    return browserPrefersDark();
  });

  const [showGuide, setShowGuide] = useState(false);

  useEffect(() => {
    if (!hasStoredTheme) return;
    try {
      localStorage.setItem("mflTheme", deluxe ? "dark" : "light");
    } catch {
      /* localStorage unavailable */
    }
  }, [deluxe, hasStoredTheme]);

  useEffect(() => {
    if (hasStoredTheme) return;
    let media: MediaQueryList;
    try {
      media = window.matchMedia("(prefers-color-scheme: dark)");
    } catch {
      return;
    }

    const syncTheme = (event: MediaQueryListEvent) => setDeluxeState(event.matches);
    media.addEventListener("change", syncTheme);
    return () => media.removeEventListener("change", syncTheme);
  }, [hasStoredTheme]);

  const setDeluxe = useCallback((updater: (prev: boolean) => boolean) => {
    setHasStoredTheme(true);
    setDeluxeState(updater);
  }, []);

  useEffect(() => {
    localStorage.setItem("mflLang", lang);
  }, [lang]);

  const closeGuide = useCallback(() => {
    setShowGuide(false);
    try {
      localStorage.setItem("mflGuideSeen", "1");
    } catch {
      /* localStorage unavailable */
    }
  }, []);

  return {
    lang,
    setLang,
    deluxe,
    setDeluxe,
    showGuide,
    setShowGuide,
    closeGuide,
  };
}
