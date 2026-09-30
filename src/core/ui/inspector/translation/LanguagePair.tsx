import { canonicalLanguage } from "@/core/services/translation/languages";
import { FC, memo, ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { RiArrowUpDownLine } from "react-icons/ri";
import { useSnapshot } from "valtio";
import { Hint, LanguageSelect } from "./common";
import { isSameLanguage } from "./logic";

/**
 * Shared language pair of the new backends (`sourceLanguage` / `targetLanguage`, canonical BCP-47; source "" = the
 * speech language, else detected). Pickers over `TRANSLATION_LANGUAGES` with names in the UI language; `children`
 * are the engine's own notes about the pair (unsupported languages, models it needs).
 *
 * Owned by P4-F.
 */
export const LanguagePair: FC<{ children?: ReactNode }> = memo(({ children }) => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.translation.data);
  const set = (key: "sourceLanguage" | "targetLanguage", value: string) =>
    window.ApiServer.patchService("translation", (s) => {
      s.data[key] = value;
    });
  const source = data.sourceLanguage.trim();
  const target = data.targetLanguage.trim();
  const canSwap = canonicalLanguage(source) !== "" && canonicalLanguage(target) !== "" && !isSameLanguage(source, target);

  const swap = () =>
    window.ApiServer.patchService("translation", (s) => {
      const from = canonicalLanguage(s.data.sourceLanguage);
      const to = canonicalLanguage(s.data.targetLanguage);
      if (!from || !to) return;
      s.data.sourceLanguage = to;
      s.data.targetLanguage = from;
    });

  return (
    <>
      <LanguageSelect
        label="transl.field_source_language"
        value={source}
        emptyLabel={t("transl.source_auto")}
        onChange={(v) => set("sourceLanguage", v)}
      />
      <div className="flex justify-end -my-1">
        <button
          type="button"
          className="btn btn-ghost btn-xs gap-1 normal-case"
          disabled={!canSwap}
          title={t("transl.pair_swap")}
          onClick={swap}
        >
          <RiArrowUpDownLine aria-hidden /> {t("transl.pair_swap")}
        </button>
      </div>
      <LanguageSelect label="transl.field_target_language" value={target} onChange={(v) => set("targetLanguage", v)} />
      {!target && <Hint tone="warning">{t("transl.error_target_missing")}</Hint>}
      {[source, target]
        .filter((code) => code !== "" && canonicalLanguage(code) === "")
        .map((code, i) => (
          <Hint key={i} tone="warning">
            {t("transl.pair_unknown_language", { code })}
          </Hint>
        ))}
      {isSameLanguage(source, target) && <Hint tone="warning">{t("transl.pair_same")}</Hint>}
      {!source && <Hint>{t("transl.pair_source_auto_hint")}</Hint>}
      {children}
    </>
  );
});
