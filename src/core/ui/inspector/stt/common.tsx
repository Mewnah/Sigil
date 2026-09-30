import { FC, ReactNode, useMemo } from "react";
import { useTranslation } from "react-i18next";

// Shared by the STT inspector panels (P2-S0 split of inspector_stt.tsx).

const detailsClass =
  "rounded-lg border border-base-content/10 [&_summary]:cursor-pointer [&_summary]:list-none [&_summary::-webkit-details-marker]:hidden";

/** Collapsible group for settings most users never need. */
export const Advanced: FC<{ title: string; children: ReactNode }> = ({ title, children }) => (
  <details className={detailsClass}>
    <summary className="px-3 py-2 text-sm font-medium">{title}</summary>
    <div className="px-3 pb-3 pt-2 flex flex-col gap-2 border-t border-base-content/10">{children}</div>
  </details>
);

/** Language name in the UI language (`Intl.DisplayNames`), falling back to the code. */
export function useLanguageName() {
  const { i18n } = useTranslation();
  return useMemo(() => {
    let names: Intl.DisplayNames | null = null;
    try {
      names = new Intl.DisplayNames([(i18n.language || "en").replace("_", "-")], { type: "language" });
    } catch {
      names = null;
    }
    return (code: string) => {
      try {
        return names?.of(code) ?? code;
      } catch {
        return code;
      }
    };
  }, [i18n.language]);
}
