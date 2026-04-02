import { useCallback, useEffect, useState } from "react";

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

  const [deluxe, setDeluxe] = useState<boolean>(() => {
    try {
      const saved = localStorage.getItem("mflTheme");
      if (saved === "dark") return true;
      if (saved === "light") return false;
    } catch {
      /* localStorage unavailable */
    }
    return false;
  });

  const [showGuide, setShowGuide] = useState(() => {
    try {
      return !localStorage.getItem("mflGuideSeen");
    } catch {
      return false;
    }
  });

  useEffect(() => {
    localStorage.setItem("mflTheme", deluxe ? "dark" : "light");
  }, [deluxe]);

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
